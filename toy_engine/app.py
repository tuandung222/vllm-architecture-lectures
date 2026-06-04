import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from toy_engine.allocator import BlockAllocator
from toy_engine.scheduler import Request, ToyScheduler
from toy_engine.model import MockModel

# 1. Khởi tạo các cấu phần hệ thống
# Cấu hình VRAM nhỏ để dễ dàng kích hoạt cơ chế Preemption (Swap) khi chạy thử nghiệm
BLOCK_SIZE = 8       # Mỗi block chứa 8 tokens
NUM_BLOCKS = 30      # Tổng cộng 30 blocks vật lý = 240 tokens tối đa trên GPU VRAM
MAX_NUM_SEQS = 4     # Tối đa 4 request chạy song song cùng lúc trên GPU
MAX_TOKENS_STEP = 64 # Ngân sách tối đa 64 tokens xử lý trong 1 step

allocator = BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
scheduler = ToyScheduler(allocator=allocator, max_num_seqs=MAX_NUM_SEQS, max_num_scheduled_tokens=MAX_TOKENS_STEP)
model = MockModel()

# Hàng đợi trung gian nhận request từ API chuyển vào Engine
input_queue: asyncio.Queue = asyncio.Queue()
# Ánh xạ request_id -> asyncio.Queue (AsyncStream) để truyền phát token về cho client
async_streams: Dict[str, asyncio.Queue] = {}

async def engine_loop():
    """
    Vòng lặp nền mô phỏng hoạt động sinh token liên tục của GPU.
    """
    print("[Engine Loop] Khởi động vòng lặp sinh token nền...")
    while True:
        # 1. Lấy tất cả request mới từ API Queue đưa vào Waiting Queue của Scheduler
        while not input_queue.empty():
            req = input_queue.get_nowait()
            scheduler.add_request(req)
            
        # 2. Nếu không có request nào hoạt động, sleep nhẹ để tiết kiệm CPU
        if not scheduler.has_requests():
            await asyncio.sleep(0.05)
            continue
            
        # 3. Lập lịch cho bước hiện tại (Iteration-level schedule)
        scheduler_output = scheduler.schedule()
        
        if not scheduler_output.scheduled_requests:
            # Có request nhưng chưa lập lịch được (ví dụ do thiếu budget hoặc bị nghẽn)
            await asyncio.sleep(0.05)
            continue
            
        # 4. Thực thi mô hình trên GPU
        generated_tokens = await model.execute(scheduler_output)
        
        # 5. Cập nhật kết quả vào scheduler và lấy ra các request đã hoàn thành
        finished_requests = scheduler.update(scheduler_output, generated_tokens)
        
        # 6. Đẩy token mới sinh ra vào stream tương ứng của từng request
        for req_id, new_token in generated_tokens.items():
            if req_id in async_streams:
                # Trả về mã giả chữ (giả lập decode đơn giản: biến token_id thành chuỗi text)
                token_text = f" token_{new_token}"
                await async_streams[req_id].put(token_text)
                
        # 7. Gửi tín hiệu hoàn thành (None) cho các request đã kết thúc để đóng kết nối
        for req in finished_requests:
            if req.request_id in async_streams:
                await async_streams[req.request_id].put(None)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi chạy vòng lặp nền khi start server
    loop_task = asyncio.create_task(engine_loop())
    yield
    # Hủy loop khi tắt server
    loop_task.cancel()

app = FastAPI(lifespan=lifespan, title="Toy LLM Serving Engine")

class GenerationParams(BaseModel):
    prompt: str
    max_tokens: int = 20

@app.post("/generate")
async def generate(params: GenerationParams, request: FastAPIRequest):
    request_id = f"req_{str(uuid.uuid4())[:6]}"
    # Tính độ dài prompt giả lập dựa trên số lượng từ
    prompt_len = max(len(params.prompt.split()), 1)
    
    # Tạo hàng đợi bất đồng bộ (AsyncStream) cho request này
    stream = asyncio.Queue()
    async_streams[request_id] = stream
    
    # Đóng gói đối tượng Request và gửi vào input queue
    new_req = Request(
        request_id=request_id,
        prompt=params.prompt,
        prompt_len=prompt_len,
        max_tokens=params.max_tokens
    )
    await input_queue.put(new_req)
    
    async def sse_generator():
        try:
            while True:
                # Kiểm tra xem Client có chủ động ngắt kết nối không
                if await request.is_disconnected():
                    # Thực hiện Abort ngay lập tức để giải phóng bộ nhớ GPU VRAM
                    scheduler.abort(request_id)
                    break
                    
                # Đợi token tiếp theo từ Engine Loop
                token_text = await stream.get()
                if token_text is None:
                    # Gặp tín hiệu kết thúc từ Engine
                    break
                yield f"data: {token_text}\n\n"
        finally:
            # Đảm bảo xóa dọn dẹp hàng đợi stream khi kết thúc
            async_streams.pop(request_id, None)
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.get("/status")
async def get_status():
    """
    Endpoint theo dõi trạng thái hệ thống: bộ nhớ VRAM, hàng đợi Scheduler.
    """
    return {
        "vram": {
            "total_blocks": NUM_BLOCKS,
            "free_blocks": allocator.get_num_free_blocks(),
            "allocated_blocks": len(allocator.block_table),
            "block_table": allocator.block_table
        },
        "scheduler": {
            "waiting_queue": [r.request_id for r in scheduler.waiting],
            "running_queue": [
                {
                    "request_id": r.request_id,
                    "prompt_len": r.prompt_len,
                    "computed_tokens": r.num_computed_tokens,
                    "generated_tokens": r.num_generated_tokens,
                    "allocated_blocks": len(allocator.get_allocated_blocks(r.request_id))
                } for r in scheduler.running
            ],
            "swapped_queue": [r.request_id for r in scheduler.swapped]
        }
    }
