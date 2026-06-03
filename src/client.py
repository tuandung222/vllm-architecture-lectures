import asyncio
import httpx
import time

SERVER_URL = "http://127.0.0.1:8000"

# Danh sách 6 requests gửi đồng thời để kiểm tra cơ chế Continuous Batching và Preemption (Swap)
# Lưu ý: Max VRAM chỉ có 30 blocks = 240 tokens, Max concurrent seqs = 4.
TEST_REQUESTS = [
    {"prompt": "Machine learning is a subset of artificial intelligence that involves training algorithms to learn patterns", "max_tokens": 25},
    {"prompt": "Deep learning models use neural networks with many layers to extract high level features", "max_tokens": 30},
    {"prompt": "Natural language processing is concerned with the interactions between computers and human languages", "max_tokens": 20},
    {"prompt": "Computer vision teaches computers to see and understand digital images and videos", "max_tokens": 25},
    {"prompt": "Reinforcement learning trains agents to make sequences of decisions to maximize cumulative reward", "max_tokens": 15},
    {"prompt": "Large language models have revolutionized NLP tasks through massive pretraining and alignment", "max_tokens": 30},
]

async def send_request(client: httpx.AsyncClient, req_index: int, params: dict, abort_after_steps: int = None):
    """
    Gửi request lên API server và stream kết quả về.
    Nếu abort_after_steps được thiết lập, client sẽ chủ động ngắt kết nối sau khi nhận được bấy nhiêu tokens.
    """
    prompt = params["prompt"]
    max_tokens = params["max_tokens"]
    
    url = f"{SERVER_URL}/generate"
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    
    start_time = time.time()
    tokens_received = 0
    req_id = None
    
    try:
        # Gửi request và giữ kết nối mở để nhận stream (SSE)
        async with client.stream("POST", url, json=payload, timeout=60.0) as response:
            if response.status_code != 200:
                print(f"[Client {req_index}] Lỗi: HTTP {response.status_code}")
                return
                
            # Đọc từng dòng trả về từ Server-Sent Events
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    token = line.replace("data: ", "").strip()
                    tokens_received += 1
                    
                    if tokens_received == 1:
                        # Ghi nhận thời gian nhận được token đầu tiên (TTFT - Time To First Token)
                        ttft = (time.time() - start_time) * 1000
                        print(f"[Client {req_index}] 🚀 Nhận token đầu tiên (TTFT: {ttft:.1f}ms): '{token}'")
                    else:
                        print(f"[Client {req_index}] Token {tokens_received}: '{token}'")
                        
                    # Mô phỏng kịch bản Client hủy kết nối đột ngột (Abort)
                    if abort_after_steps and tokens_received >= abort_after_steps:
                        print(f"[Client {req_index}] 🛑 Chủ động ngắt kết nối giữa chừng (Simulated Abort)!")
                        break
                        
            duration = time.time() - start_time
            tpot = (duration * 1000 / tokens_received) if tokens_received > 0 else 0
            print(f"[Client {req_index}] ✅ Hoàn thành! Đã nhận {tokens_received} tokens | Tổng thời gian: {duration:.2f}s | Trung bình TPOT: {tpot:.1f}ms")
            
    except Exception as e:
        print(f"[Client {req_index}] Gặp lỗi kết nối: {e}")

async def monitor_status():
    """
    Tác vụ chạy song song để liên tục theo dõi và in ra trạng thái bộ nhớ VRAM & hàng đợi lập lịch.
    """
    async with httpx.AsyncClient() as client:
        # Chạy kiểm tra trong 15 giây
        for _ in range(35):
            await asyncio.sleep(0.4)
            try:
                res = await client.get(f"{SERVER_URL}/status")
                if res.status_code == 200:
                    status = res.json()
                    vram = status["vram"]
                    sched = status["scheduler"]
                    print(f"\n--- [MONITOR] VRAM Free Blocks: {vram['free_blocks']}/{vram['total_blocks']} | Seqs Running: {len(sched['running_queue'])} | Waiting: {len(sched['waiting_queue'])} | Swapped: {len(sched['swapped_queue'])} ---")
                    if sched['running_queue']:
                        for r in sched['running_queue']:
                            print(f"  * Req {r['request_id']}: Gen {r['generated_tokens']}/{r['prompt_len'] + r['generated_tokens']} tokens (Blocks: {r['allocated_blocks']})")
                    if sched['swapped_queue']:
                        print(f"  * Request bị SWAP (lưu tại CPU RAM): {sched['swapped_queue']}")
                    print("----------------------------------------------------------------------------------------------------\n")
            except Exception:
                pass

async def main():
    print("=== Khởi động Client kiểm thử Toy Serving Engine ===")
    print(f"1. Gửi đồng thời {len(TEST_REQUESTS)} requests lên API server.")
    print("2. Mô phỏng Client số 2 (Index 2) sẽ hủy kết nối đột ngột sau khi nhận được 3 tokens.")
    print("3. Monitor song song sẽ in ra trạng thái VRAM, Waiting Queue và Swapped Queue mỗi 0.4 giây.")
    print("====================================================\n")
    
    # Khởi chạy Monitor trạng thái chạy nền
    monitor_task = asyncio.create_task(monitor_status())
    
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=10)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = []
        for i, req in enumerate(TEST_REQUESTS):
            # Client số 2 (Index 2) sẽ bị hủy kết nối sau 3 tokens để test tính năng Abort
            abort_steps = 3 if i == 2 else None
            # Trì hoãn nhẹ 50ms giữa các client gửi để tạo thứ tự đến rõ ràng
            await asyncio.sleep(0.05)
            tasks.append(send_request(client, i, req, abort_after_steps=abort_steps))
            
        await asyncio.gather(*tasks)
        
    # Chờ monitor hoàn thành
    await asyncio.sleep(1)
    monitor_task.cancel()
    print("\n=== Kết thúc kịch bản kiểm thử ===")

if __name__ == "__main__":
    asyncio.run(main())
