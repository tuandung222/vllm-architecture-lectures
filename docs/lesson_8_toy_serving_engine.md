# Bài 8: Thiết kế hệ thống Toy Serving Engine tối giản

Chào mừng bạn đến với bài học cuối cùng! Ở các bài học trước, chúng ta đã tích lũy đầy đủ kiến thức lý thuyết về cách vLLM hoạt động. Để chuyển hóa lý thuyết thành kỹ năng thực tiễn, trong bài này chúng ta sẽ cùng nhau thiết kế và hiện thực một **Toy Serving Engine** tối giản viết bằng Python.

Serving Engine này sẽ mô phỏng lại 4 tính năng quan trọng nhất của vLLM:
1. **Phân bổ KV Cache dạng trang (Paged Block Allocation)**: Quản lý các block vật lý và bảng ánh xạ trang.
2. ** Continuous Batching**: Lập lịch ở mức Iteration, tự động chèn thêm request mới và giải phóng sớm các request hoàn thành.
3. **Async Serving & Concurrency**: Sử dụng FastAPI và `asyncio` để chạy vòng lặp engine chạy song song với các request HTTP đồng thời.
4. **Streaming SSE & Abort**: Trả về token ngay khi sinh ra và giải phóng bộ nhớ lập tức nếu người dùng hủy kết nối mạng.

---

## 1. Thiết kế Kiến trúc Hệ thống

Hệ thống được chia thành 5 file mã nguồn đặt trong thư mục `src/`:

```
┌────────────────────────────────────────────────────────┐
│                   src/app.py (FastAPI)                 │
│  - Endpoint: /generate                                 │
│  - Nhận HTTP Requests -> Đưa vào asyncio.Queue          │
│  - Đọc từ AsyncStream -> Gửi stream token về Client    │
└───────────────────────────┬────────────────────────────┘
                            │ (asyncio.Queue & AsyncStream)
┌───────────────────────────▼────────────────────────────┐
│                    Async Engine Loop                   │
│  - Chạy nền trong FastAPI                              │
│  - Mỗi bước lặp: Gọi Scheduler + Chạy Model -> Output  │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │  src/scheduler.py│        │   src/model.py   │
    │  - ToyScheduler  │        │  - MockModel     │
    │  - allocator.py  │        │  - Simulates GPU │
    └──────────────────┘        └──────────────────┘
```

### Các File Mã nguồn trong thư mục `src/`:
1. **[allocator.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/src/allocator.py)**: Chứa lớp `BlockAllocator` quản lý danh sách các khối vật lý GPU trống (`free_blocks`) và bảng ánh xạ trang (`block_table`) cho từng request.
2. **[scheduler.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/src/scheduler.py)**: Chứa lớp `ToyScheduler` quản lý hàng đợi chờ (`waiting`) và hàng đợi chạy (`running`), thực hiện lập lịch continuous batching dựa trên số lượng block còn trống.
3. **[model.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/src/model.py)**: Chứa lớp `MockModel` giả lập quá trình forward của mô hình Transformer trên GPU với độ trễ (latency) ngẫu nhiên để mô phỏng pha Prefill và Decode thực tế.
4. **[app.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/src/app.py)**: Cầu nối FastAPI kết nối các cấu phần trên thành API Server bất đồng bộ hỗ trợ Streaming qua Server-Sent Events (SSE).
5. **[client.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/src/client.py)**: Kịch bản kiểm thử (test client) gửi đồng thời nhiều yêu cầu với các độ dài và tốc độ khác nhau để kiểm tra tính đúng đắn của hệ thống.

---

## 2. Thiết kế các Cấu trúc Dữ liệu chính

Để theo dõi trạng thái, chúng ta định nghĩa hai cấu trúc dữ liệu chính bằng Python dataclass:

### Cấu trúc Request:
Theo dõi trạng thái của mỗi yêu cầu gửi lên hệ thống.
```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Request:
    request_id: str
    prompt: str
    prompt_len: int
    max_tokens: int
    num_computed_tokens: int = 0
    num_generated_tokens: int = 0
    tokens: List[int] = field(default_factory=list)
    status: str = "WAITING"  # WAITING, RUNNING, FINISHED, ABORTED
```

### Cấu trúc SchedulerOutput:
Chứa danh sách các request được lập lịch chạy ở bước tiếp theo cùng thông tin số lượng token cần xử lý.
```python
@dataclass
class SchedulerOutput:
    scheduled_requests: List[Request]
    total_tokens_to_process: int
```

---

## 3. Nguyên lý Hoạt động của Async Engine Loop

Khi FastAPI khởi chạy, một tác vụ chạy nền (`background task`) được kích hoạt để chạy vòng lặp bất đồng bộ của Engine liên tục:

```python
async def engine_loop():
    while True:
        # 1. Nhận các request mới từ FastAPI thông qua asyncio.Queue chuyển vào Waiting Queue của Scheduler
        process_incoming_requests()
        
        # 2. Nếu Scheduler không có request nào, yield quyền kiểm soát cho CPU nghỉ ngơi
        if not scheduler.has_requests():
            await asyncio.sleep(0.01)
            continue
            
        # 3. Lập lịch cho bước hiện tại
        scheduler_output = scheduler.schedule()
        
        # 4. Chạy mô hình giả lập (Non-blocking bằng cách sử dụng await asyncio.to_thread)
        outputs = await model.execute(scheduler_output)
        
        # 5. Cập nhật kết quả vào Scheduler
        finished_requests = scheduler.update(scheduler_output, outputs)
        
        # 6. Đẩy các token vừa sinh ra vào AsyncStream của từng Request để FastAPI gửi về cho Client
        for req_id, new_token in outputs.items():
            if req_id in async_streams:
                await async_streams[req_id].put(new_token)
                
        # 7. Giải phóng và đóng các stream đã hoàn thành
        for req in finished_requests:
            if req.request_id in async_streams:
                await async_streams[req.request_id].put(None)  # Sentinel báo kết thúc
```

---

## 4. Cơ chế Abort khi Client Disconnect

Để xử lý ngắt kết nối:
1. Trong FastAPI endpoint, chúng ta sử dụng một vòng lặp kiểm tra trạng thái kết nối mạng:
   ```python
   @app.post("/generate")
   async def generate_endpoint(request: fastapi.Request, prompt: str, max_tokens: int):
       request_id = str(uuid.uuid4())
       # Đăng ký stream
       stream = asyncio.Queue()
       async_streams[request_id] = stream
       
       # Gửi request vào hàng đợi của Engine
       await input_queue.put(Request(request_id, prompt, ...))
       
       async def event_generator():
           try:
               while True:
                   if await request.is_disconnected():
                       # Phát hiện client ngắt kết nối
                       scheduler.abort(request_id)
                       break
                   token = await stream.get()
                   if token is None:
                       break
                   yield f"data: {token}\n\n"
           finally:
               async_streams.pop(request_id, None)
               
       return StreamingResponse(event_generator(), media_type="text/event-stream")
   ```
2. Khi `scheduler.abort(request_id)` được gọi:
   * Hệ thống tìm request trong `waiting` hoặc `running`.
   * Chuyển trạng thái của request thành `ABORTED`.
   * Gọi `BlockAllocator` giải phóng toàn bộ các block vật lý đã cấp phát cho request đó ngay lập tức.
   * Xóa request khỏi hàng đợi chạy để GPU không tốn tài nguyên xử lý ở bước tiếp theo.

---

## 🚀 Tiến hành Hiện thực mã nguồn

Chúng ta đã có thiết kế chi tiết cho toàn bộ hệ thống Toy Serving Engine. Hãy chuyển sang bước viết mã nguồn thực tế cho từng file trong thư mục `src/`!
