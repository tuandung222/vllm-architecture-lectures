import asyncio
import random
import time
from typing import Dict
from src.scheduler import SchedulerOutput

class MockModel:
    def __init__(self):
        # Bộ từ vựng giả lập
        self.eos_token_id = 50256

    async def execute(self, scheduler_output: SchedulerOutput) -> Dict[str, int]:
        """
        Giả lập quá trình chạy forward của mô hình trên GPU.
        Tính toán thời gian chạy batch thực tế và sinh token ngẫu nhiên.
        """
        if not scheduler_output.scheduled_requests:
            return {}

        # Phân tích xem trong batch có request nào đang chạy prefill không
        has_prefill = False
        max_prefill_tokens = 0
        num_decode_requests = 0
        
        for req in scheduler_output.scheduled_requests:
            tokens_to_process = scheduler_output.request_tokens_to_process[req.request_id]
            if tokens_to_process > 1:
                has_prefill = True
                max_prefill_tokens = max(max_prefill_tokens, tokens_to_process)
            else:
                num_decode_requests += 1

        # Mô hình hóa độ trễ tính toán của GPU:
        # - Pha Prefill: Compute-bound, phụ thuộc vào số lượng token đầu vào (GEMM).
        # - Pha Decode: Memory-bound, ít phụ thuộc vào batch size hơn nhưng tốn thời gian cố định nạp weight.
        if has_prefill:
            # Prefill tốn thời gian tỷ lệ thuận với prompt dài nhất
            latency = 0.05 + (0.0005 * max_prefill_tokens)
            phase_name = "PREFILL"
        else:
            # Decode tốn thời gian cố định nhỏ + tăng rất nhẹ theo batch size
            batch_size = len(scheduler_output.scheduled_requests)
            latency = 0.015 + (0.0002 * batch_size)
            phase_name = "DECODE"

        # Giả lập thời gian GPU chạy
        await asyncio.sleep(latency)

        # Sinh token output cho từng request
        outputs: Dict[str, int] = {}
        for req in scheduler_output.scheduled_requests:
            tokens_processed = scheduler_output.request_tokens_to_process[req.request_id]
            
            # Nếu đang prefill, chúng ta không sinh ra từ mới thực tế ngay (hoặc sinh từ đầu tiên)
            # vLLM: pha prefill xử lý prompt và sinh ra token output đầu tiên.
            # Để mô phỏng, prefill hay decode đều sinh ra 1 token mới ở bước này.
            
            # Xác suất kết thúc sớm (sinh ra EOS) để test dynamic completion
            # Chỉ cho phép ra EOS khi đã sinh tối thiểu 5 tokens
            if req.num_generated_tokens >= 5 and random.random() < 0.08:
                new_token = self.eos_token_id
            else:
                new_token = random.randint(1000, 20000)
                
            outputs[req.request_id] = new_token
            
        print(f"[Model GPU] Chạy {phase_name} | Batch size: {len(scheduler_output.scheduled_requests)} | Latency: {latency*1000:.2f}ms")
        return outputs
