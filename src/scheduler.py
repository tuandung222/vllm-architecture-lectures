import time
from dataclasses import dataclass, field
from typing import Dict, List
from src.allocator import BlockAllocator

@dataclass
class Request:
    request_id: str
    prompt: str
    prompt_len: int
    max_tokens: int
    num_computed_tokens: int = 0
    num_generated_tokens: int = 0
    tokens: List[int] = field(default_factory=list)
    status: str = "WAITING"  # WAITING, RUNNING, SWAPPED, FINISHED, ABORTED
    arrival_time: float = field(default_factory=time.time)

@dataclass
class SchedulerOutput:
    scheduled_requests: List[Request]
    # Bản đồ để phân biệt request đang chạy prefill (cần n tokens) hay decode (cần 1 token)
    request_tokens_to_process: Dict[str, int]
    total_tokens_to_process: int

class ToyScheduler:
    def __init__(self, allocator: BlockAllocator, max_num_seqs: int, max_num_scheduled_tokens: int):
        self.allocator = allocator
        self.max_num_seqs = max_num_seqs
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.swapped: List[Request] = []

    def add_request(self, request: Request):
        self.waiting.append(request)
        print(f"[Scheduler] Đã nhận Request {request.request_id} (Prompt len: {request.prompt_len})")

    def has_requests(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0 or len(self.swapped) > 0

    def schedule(self) -> SchedulerOutput:
        scheduled_requests: List[Request] = []
        request_tokens_to_process: Dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        
        # 1. Ưu tiên chạy các request đang RUNNING (Decode phase)
        preempted_requests = []
        for req in list(self.running):
            if token_budget <= 0:
                break
                
            # Mỗi bước decode chỉ cần sinh thêm 1 token
            num_new_tokens = 1
            
            # Yêu cầu allocator cấp phát thêm (hoặc kiểm tra xem block hiện tại còn chỗ không)
            success = self.allocator.allocate_slots(req.request_id, req.num_computed_tokens, num_new_tokens)
            
            if success:
                scheduled_requests.append(req)
                request_tokens_to_process[req.request_id] = num_new_tokens
                token_budget -= num_new_tokens
            else:
                # Cạn kiệt bộ nhớ -> Preempt request này bằng phương pháp Swapping
                print(f"[Scheduler] ⚠️ Cạn kiệt VRAM! Đang Preempt (Swap out) Request {req.request_id}")
                self.running.remove(req)
                req.status = "SWAPPED"
                self.allocator.free(req.request_id) # Giải phóng bộ nhớ GPU
                self.swapped.append(req)
                
        # 2. Xử lý phục hồi các request bị SWAPPED trước khi nhận request mới
        for req in list(self.swapped):
            if len(self.running) >= self.max_num_seqs:
                break
            # Phục hồi cần nạp lại toàn bộ KV Cache tính tới hiện tại
            tokens_to_restore = req.num_computed_tokens
            if token_budget >= tokens_to_restore:
                success = self.allocator.allocate_slots(req.request_id, 0, tokens_to_restore)
                if success:
                    print(f"[Scheduler] 🔄 Khôi phục (Swap in) Request {req.request_id} từ CPU RAM về GPU")
                    self.swapped.remove(req)
                    req.status = "RUNNING"
                    self.running.append(req)
                    # Chạy bước Decode tiếp theo
                    scheduled_requests.append(req)
                    request_tokens_to_process[req.request_id] = 1
                    token_budget -= 1

        # 3. Lập lịch cho các request đang WAITING (Prefill phase)
        for req in list(self.waiting):
            if len(self.running) >= self.max_num_seqs:
                break
                
            # Một request prefill cần xử lý toàn bộ prompt tokens cùng lúc
            needed_tokens = req.prompt_len
            
            if needed_tokens <= token_budget:
                success = self.allocator.allocate_slots(req.request_id, 0, needed_tokens)
                if success:
                    self.waiting.remove(req)
                    req.status = "RUNNING"
                    self.running.append(req)
                    scheduled_requests.append(req)
                    request_tokens_to_process[req.request_id] = needed_tokens
                    token_budget -= needed_tokens
                else:
                    # Không đủ block cho prompt của request mới -> Đợi ở vòng tiếp theo
                    break
            else:
                # Vượt quá ngân sách token tối đa của batch -> Đợi ở vòng tiếp theo
                break
                
        total_tokens = sum(request_tokens_to_process.values())
        return SchedulerOutput(
            scheduled_requests=scheduled_requests,
            request_tokens_to_process=request_tokens_to_process,
            total_tokens_to_process=total_tokens
        )

    def update(self, scheduler_output: SchedulerOutput, generated_tokens: Dict[str, int]) -> List[Request]:
        """
        Cập nhật trạng thái các request sau bước forward của mô hình.
        Trả về danh sách các request đã hoàn thành.
        """
        finished_requests: List[Request] = []
        
        for req in scheduler_output.scheduled_requests:
            # Xác định xem request vừa chạy prefill hay decode
            tokens_processed = scheduler_output.request_tokens_to_process[req.request_id]
            
            if req.num_computed_tokens == 0:
                # Vừa hoàn thành Prefill
                req.num_computed_tokens = tokens_processed
            else:
                # Vừa hoàn thành Decode thêm 1 bước
                req.num_computed_tokens += 1
                
            # Nạp token mới sinh ra
            new_token = generated_tokens.get(req.request_id)
            if new_token is not None:
                req.tokens.append(new_token)
                req.num_generated_tokens += 1
                
            # Điều kiện dừng: đạt max_tokens hoặc gặp token đặc biệt (mô phỏng EOS = 50256)
            is_eos = (new_token == 50256)
            is_max = (req.num_generated_tokens >= req.max_tokens)
            
            if is_eos or is_max:
                req.status = "FINISHED"
                reason = "EOS" if is_eos else "MAX_TOKENS"
                print(f"[Scheduler] Request {req.request_id} hoàn thành ({reason}) sau {req.num_generated_tokens} tokens.")
                if req in self.running:
                    self.running.remove(req)
                self.allocator.free(req.request_id) # Giải phóng bộ nhớ
                finished_requests.append(req)
                
        return finished_requests

    def abort(self, request_id: str):
        """
        Hủy bỏ lập lịch cho một request (khi client ngắt kết nối).
        Giải phóng bộ nhớ lập tức.
        """
        print(f"[Scheduler] 🛑 Hủy bỏ (Abort) Request {request_id} theo yêu cầu.")
        
        target_req = None
        # Tìm trong các hàng đợi
        for req in self.waiting:
            if req.request_id == request_id:
                target_req = req
                self.waiting.remove(req)
                break
                
        if not target_req:
            for req in self.running:
                if req.request_id == request_id:
                    target_req = req
                    self.running.remove(req)
                    break
                    
        if not target_req:
            for req in self.swapped:
                if req.request_id == request_id:
                    target_req = req
                    self.swapped.remove(req)
                    break
                    
        if target_req:
            target_req.status = "ABORTED"
            
        # Giải phóng bộ nhớ
        self.allocator.free(request_id)
