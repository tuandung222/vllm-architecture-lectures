---
sidebar_position: 14.0
sidebar_label: "Bài 10: Quản lý VRAM & Chiến lược Lập lịch Offline Batching"
---

# Bài 10: Quản lý VRAM & Chiến lược Lập lịch cho Xử lý Dữ liệu Hàng loạt (Offline Batching)

Khi triển khai các mô hình ngôn ngữ lớn (LLM) và mô hình đa phương thức (VLM) trong thực tế, hai vấn đề lớn nhất mà các kỹ sư thường gặp phải là:
1. Hệ thống bị sập do lỗi tràn bộ nhớ GPU (CUDA Out-Of-Memory - OOM) một cách ngẫu nhiên.
2. Không biết cách thiết lập các tham số lập lịch (Scheduling) và mức độ đồng thời (Concurrency) để đạt được băng thông xử lý (Throughput) tối đa khi xử lý dữ liệu hàng loạt.

Bài học này sẽ đi sâu phân tích cơ chế quản lý bộ nhớ VRAM tĩnh của vLLM, cách thức hoạt động của bộ lập lịch (Scheduler) trong việc điều phối các requests, và 4 chuyên đề phân tích sâu (Deep Dive) về bản chất kỹ thuật của các cơ chế tối ưu hóa cốt lõi.

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

## 4. Phân Tích Sâu (Deep Dive) 4 Cơ Chế Tối Ưu Hóa Cốt Lõi

Để làm chủ vLLM trong sản xuất, việc hiểu sơ qua định nghĩa là chưa đủ. Dưới đây là phân tích chi tiết về bản chất vận hành ở mức phần cứng và mã nguồn của 4 tính năng quan trọng nhất.

### Chuyên đề A. Bản chất kỹ thuật của `--enable-prefix-caching` (Cấu trúc Radix Tree)

Prefix Caching cho phép vLLM tái sử dụng KV Cache của các phần prompt trùng lặp (ví dụ: system prompt cố định). Bản chất của cơ chế này là **Radix Tree (Cây tiền tố)**.

#### 1. Nguyên lý so khớp mã Token và Bảng trang (Page Table)
Khi một request mới gửi đến, vLLM chuyển đổi văn bản thành một chuỗi token IDs. Bộ quản lý bộ nhớ (`kv_cache_manager.py`) sẽ băm (hash) chuỗi token IDs này theo từng khối (block size, ví dụ: 16 tokens).

Mỗi nút trên Radix Tree đại diện cho một khối token kèm theo hash của nó. Nếu request mới có các khối token đầu tiên khớp với một nhánh trên Radix Tree, vLLM sẽ thực hiện **Prefix Hit**:
- Trỏ trực tiếp Page Table của request mới vào các khối vật lý tương ứng của nhánh đó trên GPU.
- Tránh hoàn toàn việc phải chạy qua GPU forward pass để tính toán lại giá trị Key và Value của các tokens đó (tiết kiệm 100% chi phí tính toán prefill của phần trùng lặp).

```
[Mô tả Radix Tree trong vLLM]

       Root (Chung)
        │
     [System Prompt Hash] (ref_count = 2)
      /                \
  [RAG Doc A]       [RAG Doc B]
  (ref_count=1)     (ref_count=1)
```

#### 2. Cơ chế đếm tham chiếu (Reference Counting) và giải phóng LRU
Khi nhiều request chia sẻ cùng một tiền tố, bộ quản lý block sẽ tăng biến `ref_count` (số lượng tham chiếu) của các block vật lý đó lên.
- Chừng nào `ref_count > 0`, block vật lý đó được khóa cứng trên GPU và không bao giờ bị xóa.
- Khi một request hoàn thành, `ref_count` của các block riêng giảm về 0. Lúc này, thay vì giải phóng ngay lập tức về bể chứa (pool), các block này được đưa vào một danh sách **LRU (Least Recently Used) Eviction Queue**.
- Nếu GPU sắp cạn bộ nhớ và cần cấp phát block mới, vLLM sẽ tìm các block ở đầu hàng đợi LRU (ít được dùng nhất gần đây) để thu hồi và xóa dữ liệu. Nếu có request mới khớp lại các block trong LRU trước khi chúng bị xóa, block đó được nhấc ra khỏi hàng đợi LRU, tăng `ref_count` trở lại và tái sử dụng (Cache Hit).

---

### Chuyên đề B. Bản chất kỹ thuật của `--enable-chunked-prefill` (Giải quyết nghẽn Compute vs Memory)

Trong LLM serving, hai pha tính toán có đặc tính phần cứng hoàn toàn trái ngược nhau:
- **Prefill Phase:** Xử lý toàn bộ prompt đầu vào. GPU thực hiện phép nhân ma trận lớn (GEMM), cần năng lực tính toán cực cao để song song hóa. Đây là tác vụ **Compute-bound** (tốc độ bị giới hạn bởi hiệu năng tính toán của nhân Tensor Core).
- **Decode Phase:** Sinh từng token mới một. GPU thực hiện phép nhân ma trận-vector nhỏ (GEMV) nhưng phải nạp đi nạp lại toàn bộ trọng số mô hình từ bộ nhớ HBM sang bộ nhớ cache SRAM tại mỗi bước. Đây là tác vụ **Memory-bound** (tốc độ bị giới hạn bởi băng thông bộ nhớ GPU).

#### 1. Vấn đề "Lag cục bộ" (Prefill Stall) khi không bật Chunked Prefill
Khi một request mới với prompt siêu dài (ví dụ: 8K tokens) nạp vào batch đang decode của 100 người dùng khác:
- GPU phải dồn toàn lực thực hiện forward pass GEMM khổng lồ cho prompt 8K này.
- Bước forward pass này có thể mất tới $500$ ms đến $1$ giây.
- Trong thời gian này, 100 người dùng đang decode phải **chờ đợi hoàn toàn**, không thể sinh thêm token mới. Điều này tạo ra một cú giật lag cực lớn về Inter-Token Latency (ITL).

#### 2. Cơ chế bẻ nhỏ và lập lịch đan xen (Chunked Prefill)
Khi bật `--enable-chunked-prefill true`, vLLM chia nhỏ prompt 8K kia thành các đoạn nhỏ cố định (ví dụ: 16 chunks có kích thước $512$ tokens).

Tại mỗi bước lặp (iteration) của GPU:
- Bộ lập lịch nạp 1 chunk ($512$ tokens prefill) đan xen với 100 tokens decode của các request đang chạy.
- GPU xử lý phối hợp cả hai tác vụ. Nhờ kích thước chunk nhỏ, thời gian forward pass mỗi bước lặp chỉ khoảng $30 - 50$ ms.
- 100 người dùng đang decode vẫn nhận được token mới đều đặn sau mỗi vài chục mili giây, triệt tiêu hoàn toàn hiện tượng khựng lag.

```
[Lập lịch đan xen với Chunked Prefill]

Step 1: [Prefill Chunk 1 (512 tokens)] + [Decode 100 tokens] -> Chạy mất 40ms
Step 2: [Prefill Chunk 2 (512 tokens)] + [Decode 100 tokens] -> Chạy mất 40ms
...
Step 16: [Prefill Chunk 16 (512 tokens)] + [Decode 100 tokens] -> Hoàn thành prefill
```

---

### Chuyên đề C. Bản chất kỹ thuật của `--swap-space` (Hoán vị GPU-CPU và Asynchronous Copy)

Khi hệ thống bị quá tải tạm thời (ví dụ: nhiều người dùng cùng lúc sinh ra câu trả lời rất dài và GPU hết sạch block KV Cache trống), vLLM sử dụng cơ chế hoán vị (Swapping) thay vì crash OOM.

#### 1. Tại sao sử dụng bộ nhớ CPU làm vùng đệm Swap?
CPU RAM có dung lượng lớn hơn GPU VRAM rất nhiều và có giá thành rẻ hơn. vLLM đăng ký trước một vùng nhớ RAM trên CPU (mặc định 4 GiB qua `--swap-space`) và tổ chức nó thành các block có kích thước giống hệt block trên GPU.

#### 2. Cơ chế Asynchronous Copy qua CUDA Streams
Khi bộ lập lịch nhận thấy không thể cấp phát thêm block cho các request đang chạy trong danh sách `running`:
- Nó chọn ra các request có độ ưu tiên thấp nhất và đưa vào trạng thái `swapped`.
- vLLM ra lệnh cho GPU Worker di chuyển toàn bộ các block KV Cache của request đó từ GPU VRAM sang CPU RAM thông qua băng thông PCIe.
- Quá trình di chuyển này được thực hiện bất đồng bộ (Asynchronous Memory Copy) bằng các nhân CUDA non-blocking streams. Tức là luồng tính toán chính của GPU vẫn tiếp tục decode cho các request khác, trong khi luồng truyền dữ liệu PCIe chạy song song ở nền để copy dữ liệu.
- Khi các request đang chạy hoàn thành và giải phóng GPU blocks, request bị swap-out sẽ được copy ngược trở lại GPU (swap-in) để tiếp tục sinh từ.

*Tác động:* Swapping giúp hệ thống chống chịu tải đỉnh (peak load) cực kỳ an toàn, nhưng vì băng thông PCIe chậm hơn nhiều so với băng thông bộ nhớ HBM nội bộ GPU, việc lạm dụng swap (thiết lập swap-space quá lớn và chạy quá tải liên tục) sẽ làm tăng đáng kể độ trễ phản hồi tổng thể của hệ thống.

---

### Chuyên đề D. Bản chất kỹ thuật của `--enforce-eager` (CUDA Graphs vs Eager Mode)

#### 1. CPU Launch Overhead trong Deep Learning
Trong quá trình giải mã (Decode), GPU tính toán cực kỳ nhanh (chỉ mất vài mili giây cho mỗi bước). Tuy nhiên, tại mỗi bước lặp, CPU của máy chủ phải chuẩn bị các tham số, gọi các thư viện PyTorch, và gửi các lệnh thực thi (kernels launch) xuống GPU thông qua driver CUDA.

Khi thời gian chuẩn bị lệnh của CPU lớn hơn thời gian thực thi của GPU, hệ thống rơi vào trạng thái nghẽn cổ chai do CPU (**CPU-bound launch overhead**). CPU hoạt động 100% nhưng GPU liên tục phải nằm chờ lệnh, làm sụt giảm hiệu năng.

#### 2. Giải pháp CUDA Graphs (Capture & Replay) và cái giá phải trả về VRAM
CUDA Graphs giải quyết vấn đề này bằng cách chạy thử mô hình lúc khởi động, "chụp" (capture) lại toàn bộ chuỗi lệnh gọi nhân đồ thị của GPU và lưu thành một đồ thị tĩnh (static graph) duy nhất.
- Tại mỗi bước lặp tiếp theo, CPU chỉ cần phát một lệnh chạy đồ thị tĩnh này duy nhất ("replay"). Loại bỏ hoàn toàn overhead gọi lệnh của CPU.
- **Cái giá phải trả:** CUDA Graphs yêu cầu kích thước tensor đầu vào phải cố định. Để làm được điều này, vLLM phải tạo ra các nhóm kích thước (Shape Buckets) và phân bổ sẵn các buffer bộ nhớ tĩnh cho từng bucket. Các buffer đệm này chiếm một lượng VRAM tĩnh rất lớn (thường từ $1$ đến $2$ GiB).

#### 3. Khi nào nên bật Eager Mode (`--enforce-eager true`)?
Nếu bạn đặt `--enforce-eager true`, vLLM sẽ tắt hoàn toàn tính năng CUDA Graphs và thực thi PyTorch trực tiếp từng dòng (Eager Mode):
- **Ưu điểm:** Lấy lại ngay lập tức $1 - 2$ GiB VRAM tĩnh bị chiếm dụng bởi CUDA Graphs để chuyển sang làm blocks KV Cache, giúp tăng thêm số lượng câu xử lý đồng thời trên các GPU có bộ nhớ nhỏ (như L4, T4 hoặc các dòng card RTX dân dụng).
- **Nhược điểm:** Tốc độ suy luận (decode latency) của hệ thống sẽ bị chậm đi từ $10\% - 30\%$ do phải chịu launch overhead của CPU tại mỗi bước.

---

## 5. Chiến lược Tối ưu hóa cho Xử lý Dữ liệu Hàng loạt (Offline Batching)

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
       gpu_memory_util_utilization=0.90,
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

## 6. Bảng tra cứu nhanh cấu hình tinh chỉnh VRAM & Concurrency

Dưới đây là các tham số CLI cốt lõi ảnh hưởng trực tiếp đến việc phân bổ tài nguyên bộ nhớ VRAM và khả năng xử lý đồng thời trong vLLM:

| Tham số CLI | Giá trị mặc định | Tác động bộ nhớ | Tác động Concurrency | Mẹo tinh chỉnh thực chiến |
| :--- | :--- | :--- | :--- | :--- |
| `--max-model-len` | Tự động (từ model config) | **Cực lớn**. Quyết định lượng KV Cache cần thiết cho mỗi request. | **Lớn**. Giảm max length giúp tăng số block KV Cache trống. | Nếu chỉ cần xử lý hội thoại ngắn (ví dụ: tối đa 2K tokens), hãy giảm cấu hình này về đúng `2048`. Tránh để mặc định 32K hay 128K của mô hình vì sẽ làm giảm số lượng blocks KV Cache vật lý được cấp phát lúc khởi động. |
| `--enable-prefix-caching` | `false` | Tiết kiệm VRAM nhờ chia sẻ chung các block KV Cache đầu vào trùng lặp. | Tăng khả năng xử lý đồng thời vượt trội khi có nhiều request chung prompt. | Luôn bật (`true`) đối với các tác vụ RAG, chatbot nhiều lượt (Multi-turn) hoặc các prompt có chung chỉ dẫn hệ thống (system prompt) dài. |
| `--enable-chunked-prefill` | `false` | Giảm đỉnh bộ nhớ Activation của các prompt dài ở pha prefill. | Giúp điều hòa tải trọng, ngăn ngừa sụt giảm đột ngột số block trống. | Luôn bật đối với ứng dụng Chat thời gian thực (Online Chat) để giữ Inter-Token Latency (ITL) ổn định, tránh lag khựng khi có người dùng gửi prompt quá dài. |
| `--swap-space` | `4` (GiB) | Đăng ký trước một phần dung lượng RAM của CPU để làm bộ nhớ swap. | Giúp duy trì các request bị quá tải bộ nhớ thay vì phải hủy bỏ để chạy lại từ đầu. | Giữ nguyên mặc định 4 GiB. Nếu RAM máy chủ của bạn rất lớn (ví dụ 256GB), bạn có thể tăng lên `16` hoặc `32` để làm đệm an toàn khi chạy các mô hình siêu lớn. |
| `--enforce-eager` | `false` | Tiết kiệm khoảng $1 - 2$ GB VRAM tĩnh do không phải lưu trữ đồ thị tính toán CUDA Graphs. | Giúp tăng nhẹ số lượng blocks KV Cache khả dụng trên các GPU cấu hình thấp. | Chỉ bật khi bạn bị giới hạn bộ nhớ cực kỳ nghiêm ngặt trên các dòng card GPU nhỏ (như RTX 3060/4060, T4) và chấp nhận tăng nhẹ độ trễ decode (latency). |
