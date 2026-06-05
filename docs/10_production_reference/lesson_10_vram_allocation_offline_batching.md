---
sidebar_position: 14.0
sidebar_label: "Bài 10: Quản lý VRAM & Chiến lược Lập lịch Offline Batching"
---

# Bài 10: Quản lý VRAM & Chiến lược Lập lịch cho Xử lý Dữ liệu Hàng loạt (Offline Batching)

Khi triển khai các mô hình ngôn ngữ lớn (LLM) và mô hình đa phương thức (VLM) trong thực tế, hai vấn đề lớn nhất mà các kỹ sư thường gặp phải là:
1. Hệ thống bị sập do lỗi tràn bộ nhớ GPU (CUDA Out-Of-Memory - OOM) một cách ngẫu nhiên.
2. Không biết cách thiết lập các tham số lập lịch (Scheduling) và mức độ đồng thời (Concurrency) để đạt được băng thông xử lý (Throughput) tối đa khi xử lý dữ liệu hàng loạt.

Bài học này sẽ đi sâu phân tích cơ chế quản lý bộ nhớ VRAM tĩnh của vLLM, cách thức hoạt động của bộ lập lịch (Scheduler) trong việc điều phối các requests, và các bước cấu hình thực chiến để tối ưu hóa hiệu năng trong kịch bản Offline Batch Inference (Làm dữ liệu hàng loạt).

---

## 1. Cơ chế Quản lý VRAM Tĩnh của vLLM

Không giống như các thư viện suy luận truyền thống (như PyTorch nguyên bản hay Hugging Face Transformers) phân bổ bộ nhớ một cách động trong suốt quá trình chạy (dẫn đến nguy cơ sập OOM khi gặp prompt dài), vLLM sử dụng cơ chế **Cấp phát tĩnh (Static Memory Allocation)** ngay tại bước khởi động (Startup).

### Quy trình 5 bước cấp phát VRAM khi khởi tạo Engine:

Khi vLLM Engine khởi tạo tiến trình worker trên GPU, nó thực hiện các bước sau để thiết lập không gian bộ nhớ:

```
+-----------------------------------------------------------------------+
|  VRAM Vật lý của GPU (Total Memory)                                    |
+------------------+------------------+------------------+--------------+
| 1. Model Weights | 2. NCCL Buffers  | 3. Peak Act. Mem | 4. KV Cache  |
| (Tải Trọng số)   | (Đa GPU/Comm)    | (Giả lập Profile)| (Tĩnh Blocks)|
+------------------+------------------+------------------+--------------+
|<----------------- Bị kiểm soát bởi gpu_memory_utilization ------------>|
```

1. **Tải trọng số mô hình (Model Weights Loading):**
   vLLM tải các file trọng số của mô hình (định dạng Safetensors/Bin) trực tiếp lên GPU. Trọng số này chiếm một lượng bộ nhớ cố định gọi là $WeightsMemory$ (ví dụ: mô hình Llama-3-8B ở định dạng BF16 tốn khoảng $16$ GB VRAM).
2. **Khởi tạo môi trường phân tán (NCCL Initialization):**
   Với cấu hình đa GPU (sử dụng Tensor Parallelism hoặc Pipeline Parallelism), vLLM khởi tạo môi trường giao tiếp NCCL. Các buffer giao tiếp này được phân bổ trước trên GPU để đảm bảo tốc độ truyền tải chéo ma trận.
3. **Đăng ký bộ nhớ trần (Requested Memory Check):**
   vLLM chụp một snapshot trạng thái bộ nhớ ban đầu (`self.init_snapshot`) và tính toán dung lượng bộ nhớ tối đa mà nó được phép kiểm soát dựa trên tham số `--gpu-memory-utilization` (mặc định $0.92$):
   
   $$\text{RequestedMemory} = \text{TotalMemory} \times \text{gpu\_memory\_utilization}$$
   
   Nếu dung lượng bộ nhớ trống thực tế trên GPU nhỏ hơn con số $\text{RequestedMemory}$ yêu cầu này, vLLM sẽ ném ra lỗi `ValueError` và dừng tiến trình ngay lập tức để tránh tranh chấp tài nguyên với các ứng dụng khác.
4. **Giả lập đo bộ nhớ kích hoạt đỉnh (Peak Activation Memory Profiling):**
   vLLM chạy một forward pass giả lập (`self.model_runner.profile_run()`) sử dụng dữ liệu giả (dummy inputs). Kích thước của dữ liệu giả này được xác định bởi tham số giới hạn lập lịch tối đa là `--max-num-seqs` và `--max-num-batched-tokens`. Lượng bộ nhớ đỉnh PyTorch ghi nhận tăng thêm trong bước này chính là bộ nhớ cần thiết để lưu trữ các tensor kích hoạt (Activation Memory) trong quá trình tính toán: $ActivationMemory$.
5. **Cấp phát bộ đệm KV Cache vật lý (KV Cache Blocks Pre-allocation):**
   Dung lượng bộ nhớ còn lại sau khi đã khấu trừ trọng số mô hình và bộ nhớ kích hoạt đỉnh sẽ được dành toàn bộ cho KV Cache:
   
   $$\text{AvailableKVCacheMemory} = \text{RequestedMemory} - WeightsMemory - ActivationMemory - CUDAGraphMemory$$
   
   Từ dung lượng $\text{AvailableKVCacheMemory}$ này, vLLM chia cho kích thước vật lý của một block KV Cache (chứa 16 tokens theo mặc định) để xác định tổng số blocks khả dụng trên GPU (`num_blocks`):
   
   $$\text{num\_blocks} = \frac{\text{AvailableKVCacheMemory}}{\text{PageSizeBytes} \times \text{NumLayers}}$$
   
   vLLM thực hiện cấp phát trước toàn bộ các block này trên GPU dưới dạng một tensor khổng lồ cố định. Kể từ thời điểm này, bộ nhớ VRAM hoàn toàn phẳng (Flat) và không thay đổi trong suốt quá trình suy luận.

---

## 2. Cách vLLM Tính toán Concurrency và Giới hạn Lập lịch

### Khả năng xử lý đồng thời (Concurrency) thực tế có cố định không?
Câu trả lời là: **Không cố định.** 

vLLM kiểm soát khả năng xử lý đồng thời thông qua sự kết hợp giữa hai giới hạn:
- **Giới hạn cứng (Hard Constraint):** Tham số `--max-num-seqs` (mặc định là $256$). Bộ lập lịch (Scheduler) sẽ không bao giờ cho phép số lượng request chạy song song trong một bước lặp vượt quá con số này.
- **Giới hạn mềm động (Dynamic Soft Constraint):** Dựa trên số lượng block KV Cache trống khả dụng. Mỗi request khi sinh ra token mới (giai đoạn decode) hoặc nhận prompt đầu vào (giai đoạn prefill) sẽ yêu cầu thêm các block KV Cache. 

Nếu các request gửi lên có ngữ cảnh quá dài (ví dụ: prompt RAG dài 8K tokens), mỗi request sẽ chiếm dụng rất nhiều blocks. Khi GPU hết sạch block KV Cache trống, Scheduler bắt buộc phải giữ các request mới ở hàng đợi `waiting` cho đến khi các request cũ hoàn thành và giải phóng bộ nhớ, ngay cả khi số lượng request đang chạy thực tế mới chỉ đạt $10$ hay $20$ (dưới rất xa mức `--max-num-seqs`).

### Chuyện gì xảy ra nếu người dùng cài đặt `--max-num-seqs` quá lớn?
Khi người dùng đặt tham số `--max-num-seqs` vượt quá khả năng thực tế của GPU (ví dụ: đặt thành $1024$ hay $2048$ trên card GPU 24GB):
1. **Lỗi CUDA OOM ngay khi khởi động:** Ở bước 4 (Profiling), vLLM giả lập chạy thử với số lượng sequence khổng lồ này. Lượng activation memory tăng vọt vượt quá giới hạn vật lý của GPU, gây ra lỗi OOM lập tức tại thời điểm startup.
2. **Hiệu suất suy luận sụt giảm nghiêm trọng:** Nếu cấu hình lọt qua được bước startup, lượng bộ nhớ được PyTorch giữ chỗ cho $ActivationMemory$ quá lớn sẽ bóp nghẹt dung lượng còn lại dành cho KV Cache. Số lượng block KV Cache vật lý thực tế được cấp phát sẽ giảm đi đáng kể. Khi chạy thực tế, vLLM liên tục gặp tình trạng cạn kiệt block KV Cache, buộc Scheduler phải kích hoạt cơ chế **Preemption** (đẩy các request có độ ưu tiên thấp ra ngoài, giải phóng block của chúng sang CPU RAM hoặc bắt đầu tính toán lại từ đầu khi nạp lại) khiến tốc độ sinh từ bị gián đoạn và throughput giảm mạnh.

---

## 3. Khái niệm "Batch" trong vLLM & `max_num_batched_tokens`

### Khái niệm "Batch" động (Continuous Batching)
Trong các thư viện suy luận truyền thống, batch được xử lý ở cấp độ request (Request-level batching). Tất cả các request trong một batch phải bắt đầu và kết thúc cùng lúc. Nếu các câu hỏi có độ dài khác nhau, hệ thống phải chèn thêm các token đệm vô nghĩa (padding) để khớp kích thước, gây lãng phí năng lực tính toán của GPU.

vLLM áp dụng kỹ thuật **Continuous Batching (Lập lịch ở cấp độ bước lặp - Iteration-level scheduling)**:
- Batch ở mỗi bước lặp của GPU là một tập hợp động các token.
- Nó có thể chứa cả các token đang decode (mỗi request đóng góp 1 token tại bước đó) và các token đang prefill của các request mới được nạp vào.
- Nhờ cơ chế quản lý địa chỉ nhớ ảo PagedAttention, vLLM hoàn toàn không cần chèn các token đệm (zero padding).

```
Continuous Batching Iteration 1:
[Req 1: Decode Token #5] [Req 2: Decode Token #12] [Req 3: Prefill Token #1] ... [Req 3: Prefill Token #512]

Continuous Batching Iteration 2:
[Req 1: Decode Token #6] [Req 2: Decode Token #13] [Req 3: Decode Token #1] ... [Req 4: Prefill Token #1]
```

### Ý nghĩa của tham số `--max-num-batched-tokens`
Tham số `--max-num-batched-tokens` giới hạn tổng số lượng token (cả prefill và decode) tối đa được phép gom lại để thực hiện forward pass trong một bước lặp đơn lẻ của GPU. 

Nếu bạn gửi lên nhiều prompt mới cùng lúc và tổng số token prompt vượt quá giới hạn này, Scheduler sẽ hoãn việc nạp một số request hoặc chia nhỏ prompt đầu vào (nếu bật Chunked Prefill) để đảm bảo không vượt quá mức trần, tránh gây quá tải tính toán và giữ lượng Activation Memory trong tầm kiểm soát.

---

## 4. Chiến lược Tối ưu hóa cho Xử lý Dữ liệu Hàng loạt (Offline Batching)

Khi sử dụng vLLM cho các tác vụ xử lý dữ liệu ngoại tuyến (Offline Batching) như: gắn nhãn dữ liệu, dịch thuật hàng loạt, hoặc sinh dữ liệu tổng hợp (synthetic data generation), mục tiêu cao nhất của bạn là **băng thông tổng (Throughput - số token sinh ra trên giây trên mỗi GPU)** chứ không phải thời gian phản hồi tức thì (Latency).

### Hướng dẫn thiết lập tối ưu:

1. **Thiết lập Concurrency phía Client cực cao:**
   Để tận dụng tối đa sức mạnh của GPU, bạn phải giữ cho hàng đợi `waiting` của vLLM luôn có sẵn công việc để nạp. Phía client nên đẩy đồng thời (concurrency) từ **$500$ đến $1000$ requests** cùng lúc.
2. **Sử dụng Lập trình Không đồng bộ (Async Client):**
   Không nên tạo nhiều tiến trình (multi-processing) hoặc luồng (multi-threading) ở phía client gửi request vì sẽ gây lãng phí tài nguyên CPU của máy client và gặp overhead đồng bộ luồng. Hãy dùng một vòng lặp không đồng bộ đơn luồng (single-thread event loop) bằng các thư viện async trong Python như `asyncio` kết hợp với `httpx.AsyncClient` hoặc `aiohttp`.
3. **Loại bỏ HTTP Overhead bằng Offline API (`vllm.LLM`):**
   Nếu tiến trình xử lý dữ liệu chạy trên cùng một máy chủ vật lý chứa GPU, bạn **không nên chạy vLLM dưới dạng Server API HTTP** (cổng 8000). Hãy gọi trực tiếp thư viện vLLM trong code Python của bạn thông qua lớp `vllm.LLM`. Điều này loại bỏ hoàn toàn chi phí serialize/deserialize dữ liệu JSON và overhead truyền thông qua mạng HTTP:
   
   ```python
   import asyncio
   from vllm import LLM, SamplingParams

   # Khởi tạo engine vLLM trực tiếp trong script Python
   # pre-allocate 90% bộ nhớ GPU cho suy luận
   llm = LLM(
       model="Qwen/Qwen2.5-7B-Instruct",
       gpu_memory_utilization=0.90,
       max_model_len=4096
   )

   # Danh sách hàng nghìn prompt cần xử lý
   prompts = [
       "Hãy tóm tắt đoạn văn sau: ...",
       "Phân tích sắc thái của bình luận: ...",
       # ... hàng ngàn câu lệnh khác
   ]

   sampling_params = SamplingParams(
       temperature=0.7,
       top_p=0.9,
       max_tokens=512
   )

   # vLLM tự động gom nhóm, tối ưu hóa bộ nhớ KV Cache và chạy Continuous Batching
   # ở tốc độ phần cứng cao nhất mà không tốn chi phí mạng HTTP
   outputs = llm.generate(prompts, sampling_params)

   for output in outputs:
       prompt = output.prompt
       generated_text = output.outputs[0].text
       # Xử lý kết quả đầu ra...
   ```

---

## 5. Bảng tra cứu nhanh cấu hình tinh chỉnh VRAM & Concurrency

Dưới đây là các tham số CLI cốt lõi ảnh hưởng trực tiếp đến việc phân bổ tài nguyên bộ nhớ VRAM và khả năng xử lý đồng thời trong vLLM:

| Tham số CLI | Giá trị mặc định | Tác động bộ nhớ | Tác động Concurrency | Mẹo tinh chỉnh thực chiến |
| :--- | :--- | :--- | :--- | :--- |
| `--max-model-len` | Tự động (từ model config) | **Cực lớn**. Quyết định lượng KV Cache cần thiết cho mỗi request. | **Lớn**. Giảm max length giúp tăng số block KV Cache trống. | Nếu chỉ cần xử lý hội thoại ngắn (ví dụ: tối đa 2K tokens), hãy giảm cấu hình này về đúng `2048`. Tránh để mặc định 32K hay 128K của mô hình vì sẽ làm giảm số lượng blocks KV Cache vật lý được cấp phát lúc khởi động. |
| `--enable-prefix-caching` | `false` | Tiết kiệm VRAM nhờ chia sẻ chung các block KV Cache đầu vào trùng lặp. | Tăng khả năng xử lý đồng thời vượt trội khi có nhiều request chung prompt. | Luôn bật (`true`) đối với các tác vụ RAG, chatbot nhiều lượt (Multi-turn) hoặc các prompt có chung chỉ dẫn hệ thống (system prompt) dài. |
| `--enable-chunked-prefill` | `false` | Giảm đỉnh bộ nhớ Activation của các prompt dài ở pha prefill. | Giúp điều hòa tải trọng, ngăn ngừa sụt giảm đột ngột số block trống. | Luôn bật đối với ứng dụng Chat thời gian thực (Online Chat) để giữ Inter-Token Latency (ITL) ổn định, tránh lag khựng khi có người dùng gửi prompt quá dài. |
| `--swap-space` | `4` (GiB) | Đăng ký trước một phần dung lượng RAM của CPU để làm bộ nhớ swap. | Giúp duy trì các request bị quá tải bộ nhớ thay vì phải hủy bỏ để chạy lại từ đầu. | Giữ nguyên mặc định 4 GiB. Nếu RAM máy chủ của bạn rất lớn (ví dụ 256GB), bạn có thể tăng lên `16` hoặc `32` để làm đệm an toàn khi chạy các mô hình siêu lớn. |
| `--enforce-eager` | `false` | Tiết kiệm khoảng $1 - 2$ GB VRAM tĩnh do không phải lưu trữ đồ thị tính toán CUDA Graphs. | Giúp tăng nhẹ số lượng blocks KV Cache khả dụng trên các GPU cấu hình thấp. | Chỉ bật khi bạn bị giới hạn bộ nhớ cực kỳ nghiêm ngặt trên các dòng card GPU nhỏ (như RTX 3060/4060, T4) và chấp nhận tăng nhẹ độ trễ decode (latency). |
