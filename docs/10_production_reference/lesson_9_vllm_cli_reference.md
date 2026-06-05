---
sidebar_position: 13.0
sidebar_label: "Bài 9: Cẩm nang Tinh chỉnh CLI & Tham chiếu Production"
---

# Bài 9: Cẩm nang Tinh chỉnh CLI và Tham chiếu Production

Việc đưa các mô hình ngôn ngữ lớn (LLM) và mô hình đa phương thức (VLM) vào vận hành thực tế yêu cầu các kỹ sư phải đưa ra các quyết định cấu hình tối ưu để cân bằng giữa băng thông xử lý (Throughput), độ trễ (Latency) và chi phí phần cứng. vLLM cung cấp hàng chục tham số CLI (Command Line Interface), nhưng nếu điều chỉnh thiếu cơ sở, hệ thống có thể bị sụt giảm hiệu năng nghiêm trọng hoặc liên tục gặp lỗi tràn bộ nhớ (Out-Of-Memory - OOM).

Bài học này sẽ đóng vai trò như một cẩm nang tra cứu và tinh chỉnh hiệu năng toàn diện, phân tích chi tiết ý nghĩa kỹ thuật của từng tham số CLI cốt lõi, yêu cầu kiến thức nền tảng và các kinh nghiệm thực tế (Tuning Tips/Tricks) để làm chủ cấu hình vLLM v1.

---

## 1. Nhóm Cấu hình Mô hình & Kiểu Dữ liệu (Model Loading & Precision)

Nhóm cấu hình này xác định cách vLLM tải trọng số mô hình lên bộ nhớ GPU và sử dụng định dạng số thực nào để tính toán.

### A. `--model <path_or_repo>` & `--tokenizer <path_or_repo>`
*   **Ý nghĩa**: Đường dẫn vật lý đến thư mục chứa trọng số mô hình (định dạng HuggingFace/Safetensors) cục bộ hoặc ID repository trực tuyến.
*   **Kiến thức nền tảng**: Cách tổ chức file của HuggingFace, định dạng Safetensors (tải file zero-copy qua memory mapping), và phân biệt giữa tokenizer của LLM (chuyển text sang token ID) và model weights.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**: 
    *   Trong môi trường production, hãy luôn tải sẵn mô hình về ổ cứng SSD/NVMe cục bộ và truyền đường dẫn vật lý thay vì để server tự động tải từ HuggingFace Hub khi khởi động. Việc này giúp giảm đáng kể thời gian khởi động (cold start time) của container và tránh lỗi mạng ngắt quãng.

### B. `--dtype <format>`
*   **Ý nghĩa**: Xác định kiểu dữ liệu số thực sử dụng trong các tính toán forward pass (`auto`, `half`, `float16`, `bfloat16`, `float`).
*   **Kiến thức nền tảng**: Kiến thức về độ chính xác số học (Precision Formats) và kiến trúc GPU microarchitecture:
    *   `float16` (FP16) và `bfloat16` (BF16) đều chiếm $2$ Bytes ($16$ bits) bộ nhớ, tiết kiệm một nửa VRAM so với FP32 ($4$ Bytes).
    *   **BF16** giữ nguyên số bit cho phần mũ (exponent) giống FP32 nên có dải biểu diễn (dynamic range) tương đương FP32, giúp ngăn chặn lỗi tràn số (overflow) hoặc mất mát thông tin (underflow) khi huấn luyện và suy luận. Tuy nhiên, BF16 yêu cầu phần cứng GPU kiến trúc Ampere trở lên (A100, RTX 3090, H100). FP16 chạy được trên các GPU cũ hơn (Turing/Volta như V100, T4).
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Luôn chọn `--dtype bfloat16` nếu GPU hỗ trợ và mô hình gốc được huấn luyện bằng BF16 (hầu hết các mô hình hiện nay như Llama-3, Qwen-2). Việc ép kiểu về `float16` có thể gây ra hiện tượng sinh text rác (NaN tokens) do tràn số ở các lớp LayerNorm hoặc Attention.

### C. `--trust-remote-code`
*   **Ý nghĩa**: Cho phép thực thi mã nguồn tùy chỉnh được định nghĩa trực tiếp trong repository của mô hình trên HuggingFace.
*   **Kiến thức nền tảng**: Cơ chế nạp động module của Python (`sys.modules`), kiến trúc lớp mô hình tự định nghĩa (Custom Model Architectures) trong thư viện `transformers`.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Nhiều mô hình đa phương thức (VLM) hoặc mô hình mới ra mắt yêu cầu cờ này để chạy được Vision Tower hoặc Projector riêng biệt. Tuy nhiên, đây là kẽ hở bảo mật lớn (RCE - Remote Code Execution). 
    *   Trong môi trường production an toàn, hãy tải code mô hình về, kiểm tra (audit) file `modeling_*.py`, sau đó đóng gói mô hình cục bộ hoặc chỉ định rõ tham số `--revision` cố định để khóa phiên bản commit an toàn, tránh bị tiêm mã độc vào model repo trực tuyến.

---

## 2. Nhóm Phân tán & Song song hóa (Distributed & Parallelism)

Nhóm cấu hình này điều phối việc phân rã mô hình để chạy trên hệ thống đa GPU.

```
                  ┌────────────────────────────────────────┐
                  │          --tensor-parallel-size        │
                  │   (Chia ma trận tính toán trên GPU)    │
                  └───────────────────┬────────────────────┘
                                      ▼
                  ┌────────────────────────────────────────┐
                  │         --pipeline-parallel-size       │
                  │ (Chia các layer mô hình thành các stage)│
                  └───────────────────┬────────────────────┘
                                      ▼
                  ┌────────────────────────────────────────┐
                  │     --distributed-executor-backend     │
                  │    (Ray vs Multiprocessing Worker)     │
                  └────────────────────────────────────────┘
```

### A. `--tensor-parallel-size` (hoặc `-tp`)
*   **Ý nghĩa**: Số lượng GPU chia sẻ tính toán song song ma trận (Tensor Parallelism - TP) theo giải thuật Megatron-LM.
*   **Kiến thức nền tảng**: 
    *   **Megatron-LM TP**: Chia ma trận trọng số của Attention (Column Parallel) và MLP (Row Parallel).
    *   Mỗi bước forward pass yêu cầu $2$ lần giao tiếp **All-Reduce** qua thư viện truyền thông NCCL để đồng bộ dữ liệu giữa các GPU.
    *   Băng thông vật lý chéo: NVLink (lên tới $900$ GB/s trên H100) có tốc độ lớn hơn hàng chục lần so với PCIe Gen4/Gen5 ($32 - 64$ GB/s).
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Giá trị TP nên là ước số của số GPU trên một node vật lý và **không nên vượt quá số GPU nằm trong cùng một switch NVLink**. Ví dụ, trên server 8x A100 có NVLink toàn bộ, bạn có thể thiết lập `TP=8`. 
    *   Nếu bạn chạy trên các GPU không có NVLink kết nối trực tiếp (chỉ truyền qua PCIe), thiết lập TP lớn (như `TP=4` hoặc `TP=8`) sẽ gây nghẽn mạng nghiêm trọng do latency của All-Reduce qua PCIe cực lớn, làm sụt giảm throughput của hệ thống còn tệ hơn chạy TP nhỏ.

### B. `--pipeline-parallel-size` (hoặc `-pp`)
*   **Ý nghĩa**: Số lượng phân đoạn song song đường ống (Pipeline Parallelism - PP), cắt dọc các Layer của mô hình thành các phân đoạn (stages) chạy trên các GPU nối tiếp.
*   **Kiến thức nền tảng**: Cơ chế lập lịch đường ống (1F1B - One Forward, One Backward hoặc Forward-only), hiện tượng bong bóng đường ống (Pipeline Bubble Overhead) làm GPU rảnh rỗi chờ dữ liệu từ stage trước, và giao tiếp truyền nhận Point-to-Point (Send/Recv) qua mạng.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Chỉ sử dụng PP khi kích thước mô hình vượt quá bộ nhớ của một node vật lý (ví dụ phục vụ mô hình Llama-3 405B yêu cầu hơn $800$ GB VRAM, vượt quá dung lượng của một node 8x A100 80GB).
    *   Hãy luôn ưu tiên cấu hình tối đa TP trong một node trước (ví dụ TP=8), sau đó mới nâng PP để truyền thông tin xuyên nodes (ví dụ TP=8, PP=2 chạy trên 16 GPU ở 2 nodes). Hãy bật GPUDirect RDMA (`NCCL_IB_DISABLE=0`) để GPU truyền nhận dữ liệu trực tiếp qua card mạng InfiniBand/RoCE mà không cần đi vòng qua CPU RAM.

### C. `--distributed-executor-backend`
*   **Ý nghĩa**: Chọn backend điều phối tiến trình con (Worker processes), gồm hai lựa chọn `ray` hoặc `mp` (multiprocessing).
*   **Kiến thức nền tảng**: Kiến thức hệ điều hành về đa tiến trình (multiprocessing), cơ chế IPC (Inter-Process Communication), và kiến trúc điều phối của Ray (Ray Actor, Ray GCS).
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Trong vLLM v1, backend mặc định cho các cấu hình chạy đơn node (Single-node multi-GPU) là `mp` (multiprocessing). Lớp `MultiprocExecutor` sử dụng ZeroMQ và Shared Memory IPC cho tốc độ khởi động cực nhanh và bỏ qua được overhead kết nối của Ray.
    *   Chỉ chọn `--distributed-executor-backend ray` khi bắt buộc phải chạy phân tán đa node (Multi-node serving) vượt quá giới hạn 1 máy vật lý.

---

## 3. Nhóm Quản lý Bộ nhớ & VRAM (Memory & KV Cache)

Đây là nhóm tham số trực tiếp tác động tới tính ổn định của hệ thống phục vụ. Điều chỉnh sai lệch ở đây là nguyên nhân trực tiếp gây ra lỗi GPU OOM.

### A. `--gpu-memory-utilization`
*   **Ý nghĩa**: Tỷ lệ VRAM của GPU mà vLLM được phép chiếm dụng sau khi đã tải trọng số mô hình (mặc định $0.90$). Dung lượng còn lại sẽ được phân bổ toàn bộ cho KV Cache vật lý.
*   **Kiến thức nền tảng**: Cấu trúc bộ nhớ GPU khi phục vụ LLM:
    
    $$\text{VRAM}_{\text{Total}} = \text{VRAM}_{\text{ModelWeights}} + \text{VRAM}_{\text{KVCache}} + \text{VRAM}_{\text{ActivationMemory}}$$
    
    *   $\text{VRAM}_{\text{ModelWeights}}$: Cố định, phụ thuộc kích thước mô hình và độ chính xác (ví dụ mô hình 70B BF16 cần $140$ GB).
    *   $\text{VRAM}_{\text{KVCache}}$: vLLM tiền cấp phát (pre-allocate) một vùng nhớ lớn bằng cách tính toán số lượng blocks vật lý tối đa có thể chứa trong không gian còn lại.
    *   $\text{VRAM}_{\text{ActivationMemory}}$: Bộ nhớ đệm tạm thời phát sinh trong quá trình forward pass (tỷ lệ thuận với Batch Size và độ dài ngữ cảnh prefill).
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Nếu bạn chạy mô hình LLM thông thường với prompt ngắn và batch size nhỏ, bạn có thể tăng tham số này lên `0.92` hoặc `0.95` để tăng tối đa dung lượng KV Cache, giúp hệ thống phục vụ được nhiều request đồng thời hơn.
    *   Tuy nhiên, nếu bạn chạy các mô hình đa phương thức (VLM) có Vision Tower nặng hoặc chạy ngữ cảnh siêu dài (32K+ tokens), pha Prefill sẽ sinh ra lượng bộ nhớ kích hoạt ($\text{VRAM}_{\text{ActivationMemory}}$) khổng lồ. Nếu đặt utilization quá cao, hệ thống sẽ sập OOM lập tức ở pha prefill batch. Khi đó, bắt buộc phải hạ utilization xuống `0.80` hoặc `0.85` để dành khoảng trống (headroom) cho GPU tính toán.

### B. `--max-num-batched-tokens`
*   **Ý nghĩa**: Số lượng token tối đa được phép xử lý (forward) trong một bước lặp (iteration) của batch.
*   **Kiến thức nền tảng**: Mô hình Roofline (Roofline Model) và cường độ số học (Arithmetic Intensity):
    
    $$I = \frac{\text{Phép tính FLOPs}}{\text{Dung lượng đọc/ghi Bytes}}$$
    
    *   Khi số lượng token tính toán trong một bước forward quá nhỏ, GPU sẽ rơi vào vùng **Memory-bound** (phần cứng dành phần lớn thời gian nạp trọng số từ HBM vào SRAM, năng lực tính toán của các nhân Tensor Core bị lãng phí).
    *   Bằng cách tăng lượng token xử lý song song trong một batch, cường độ số học tăng lên, đưa hệ thống tiệm cận vùng **Compute-bound** (GPU hoạt động hết công suất tính toán toán học, đạt hiệu suất phần cứng cao nhất).
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Giá trị mặc định của cờ này thường là $2048$ hoặc tự động tính toán tương đương kích thước ngữ cảnh tối đa của mô hình. 
    *   Để tối ưu hóa thông lượng (Throughput) trong kịch bản offline batching, hãy tăng giá trị này lên tối đa bộ nhớ cho phép (ví dụ $8192$ hoặc $16384$). 
    *   Tuy nhiên, tăng giá trị này quá cao sẽ khiến thời gian xử lý pha prefill của một request dài tăng lên, gián tiếp làm khựng (stall) quá trình decode của các request khác trong batch, làm tăng Inter-Token Latency (ITL). Do đó, trong các ứng dụng chatbot thời gian thực (Online Chat), hãy giữ giá trị này ở mức vừa phải ($2048$ hoặc $4096$) kết hợp với Chunked Prefill.

### C. `--max-num-seqs`
*   **Ý nghĩa**: Số lượng request (chuỗi câu hỏi) tối đa chạy song song trong một batch hoạt động.
*   **Kiến thức nền tảng**: Khái niệm Continuous Batching và chi phí quản lý metadata của scheduler.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Nếu bạn phục vụ mô hình nhỏ (như Llama-3-8B) trên GPU mạnh (A100 80GB), tài nguyên KV Cache rất dư dả. Hãy tăng `--max-num-seqs` lên $256$ hoặc $512$ để tối đa hóa throughput của GPU.
    *   Nếu bạn phục vụ mô hình lớn (như Llama-3-70B), việc mở batch size quá lớn sẽ làm cạn kiệt block KV Cache rất nhanh, dẫn đến hiện tượng bộ lập lịch phải liên tục đẩy các request đang sinh từ ra ngoài (Preemption qua Swapping hoặc Recomputation) gây sụt giảm hiệu năng nghiêm trọng. Hãy đặt giá trị này vừa phải ($64$ hoặc $128$).

### D. `--block-size <int>`
*   **Ý nghĩa**: Kích thước khối KV Cache vật lý (đại diện cho số lượng token chứa trong 1 block bộ nhớ, mặc định là $16$).
*   **Kiến thức nền tảng**: Nguyên lý phân mảnh bộ nhớ (Memory Fragmentation):
    *   **Phân mảnh trong (Internal Fragmentation)**: Xảy ra ở block cuối cùng của request khi nó không chứa đầy token.
    *   **Overhead quản lý**: Mỗi block yêu cầu một con trỏ ánh xạ trong Page Table của Block Manager. Kích thước block càng nhỏ, số lượng block càng nhiều, gây áp lực lên tài nguyên CPU khi tra cứu bảng trang.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Hầu hết các Attention Kernel (như PagedAttention v2) được tối ưu hóa tốt nhất ở kích thước block $16$ hoặc $32$. 
    *   Không nên hạ block size xuống $8$ vì mặc dù giảm phân mảnh trong, nó sẽ làm tăng gấp đôi số lượng block cần quản lý, làm CPU bị quá tải ở luồng lập lịch Scheduler. 
    *   Cấu hình khuyến nghị: Sử dụng mặc định $16$. Với các mô hình ngữ cảnh siêu dài, có thể tăng lên $32$ để giảm dung lượng bảng trang Page Table.

---

## 4. Nhóm Tối ưu hóa Lập lịch & Latency (Scheduling Tricks)

```
       [ Client Request ] ➔ [ Request Queue ]
                                  │
                       (Chunked Prefill Enabled?)
                      /                         \
                   [YES]                        [NO]
                    /                             \
     ┌─────────────────────────────┐        ┌─────────────────────────────┐
     │ Bẻ nhỏ Prompt 8K thành      │        │ Prefill toàn bộ 8K tokens   │
     │ 16 chunks x 512 tokens      │        │ trong 1 bước forward duy nhất│
     │ Chạy xen kẽ với Decode      │        │ GPU bị compute nghẽn        │
     │ -> ITL ổn định, mượt mà     │        │ -> Decode khác bị đóng băng │
     └─────────────────────────────┘        └─────────────────────────────┘
```

### A. `--enable-chunked-prefill`
*   **Ý nghĩa**: Kích hoạt tính năng bẻ nhỏ prompt prefill siêu dài thành các mảnh nhỏ kích thước cố định để chạy đan xen với các request decode khác trong cùng một batch.
*   **Kiến thức nền tảng**: Sự bất đối xứng về hành vi phần cứng giữa Prefill (GEMM - Compute-bound) và Decode (GEMV - Memory-bound). Nếu một prefill cực lớn (ví dụ prompt RAG 8K tokens) chen vào batch, GPU phải thực hiện forward pass GEMM khổng lồ trong thời gian dài (ví dụ $500$ ms), khiến luồng Decode của tất cả người dùng khác bị đóng băng (stall), gây gián đoạn cục bộ độ trễ.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   **Luôn bật cờ này (`--enable-chunked-prefill true`) cho các hệ thống Online Serving** chạy tác vụ Chat, RAG hoặc Agent yêu cầu xử lý prompt đầu vào dài. Nó sẽ triệt tiêu hoàn toàn hiện tượng khựng lag văn bản, ổn định chỉ số Inter-Token Latency (ITL) ở mức mượt mà nhất.
    *   Tuy nhiên, bật Chunked Prefill sẽ làm tăng nhẹ tổng thời gian phản hồi token đầu tiên (TTFT) của chính request dài đó do phải chia nhỏ ra chạy nhiều lượt. Nếu bạn chạy offline batching ưu tiên throughput thô, hãy tắt tính năng này.

### B. `--enable-prefix-caching`
*   **Ý nghĩa**: Kích hoạt cơ chế tự động lưu trữ và tái sử dụng bộ đệm KV Cache của các đoạn prompt tiền tố trùng lặp (ví dụ: system prompt cố định, tài liệu ngữ cảnh RAG chung).
*   **Kiến thức nền tảng**: Cấu trúc dữ liệu cây tiền tố (Radix Tree) áp dụng vào việc đối chiếu token sequence IDs và chính sách giải phóng bộ nhớ LRU Eviction.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Bật tính năng này là **bắt buộc** nếu bạn xây dựng các ứng dụng RAG (truy vấn tài liệu chung liên tục), hội thoại nhiều lượt (Multi-turn Chat) nơi lịch sử chat liên tục được gửi lại, hoặc các ứng dụng dịch thuật/code sinh tự động có chung system prompt dài. 
    *   Nó giúp đưa Time-to-First-Token (TTFT) của các lượt chat sau về gần mức 0 ms vì hệ thống nhận diện prefix hit và bỏ qua toàn bộ pha Prefill tính toán.

---

## 5. Nhóm Speculative Decoding & Multimodal

Các tham số nâng cao để phục vụ các kịch bản suy luận hiện đại.

### A. `--speculative-model <draft_model_name>`
*   **Ý nghĩa**: Chỉ định mô hình nháp (Draft Model) kích thước nhỏ chạy song song để dự đoán trước các token nháp cho mô hình chính (Target Model).
*   **Kiến thức nền tảng**: Thuật toán Rejection Sampling kiểm chứng xác suất token nháp trên target model, cơ chế quản lý rollback KV Cache khi token bị từ chối, và overhead truyền thông tin giữa Draft Worker và Target Worker.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Mô hình nháp phải có cùng bộ từ vựng (Vocabulary) với mô hình chính. Tỷ lệ kích thước tối ưu giữa Draft và Target thường là $1:10$ hoặc $1:20$ (ví dụ dùng Llama-3-8B làm draft cho Llama-3-70B).
    *   Nếu bạn chạy trên hệ thống GPU đơn, hãy kiểm tra kỹ băng thông PCIe. Nếu mô hình nháp quá lớn, chi phí copy trạng thái KV Cache và đồng bộ tensor giữa hai mô hình có thể lớn hơn cả thời gian sinh từ trực tiếp của mô hình chính, làm sụt giảm tốc độ suy luận.

### B. `--limit-mm-per-prompt <limit_string>`
*   **Ý nghĩa**: Giới hạn số lượng đối tượng đa phương thức tối đa (ảnh, video, âm thanh) trong mỗi prompt (ví dụ: `image=2,video=1`).
*   **Kiến thức nền tảng**: Cơ chế hoạt động của Vision Tower (ViT) và Projector biến ảnh thành hàng ngàn visual tokens nhúng thẳng vào ngữ cảnh của LLM.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Luôn cấu hình cờ này trong production để bảo vệ GPU VRAM. Một hình ảnh thô có thể tương đương $576$ hoặc $1024$ tokens ngữ cảnh. 
    *   Nếu không đặt giới hạn, người dùng có thể gửi prompt chứa $10 - 20$ tấm ảnh chất lượng cao, lập tức chiếm dụng hàng chục ngàn blocks KV Cache và làm crash GPU Worker do tràn bộ nhớ vật lý.

---

## 6. Nhóm Tối ưu hóa Phần cứng & Trình biên dịch (Engine Performance)

Nhóm cấu hình can thiệp sâu vào cách GPU lập lịch thực thi các nhân tính toán (kernels).

### A. `--enforce-eager`
*   **Ý nghĩa**: Bắt buộc vLLM chạy ở chế độ Eager Mode (thực thi PyTorch trực tiếp từng dòng lệnh) thay vì sử dụng cơ chế ghi lại đồ thị tính toán CUDA Graphs.
*   **Kiến thức nền tảng**: 
    *   **CPU Launch Overhead**: Khi batch size nhỏ và mô hình chạy quá nhanh, thời gian CPU chuẩn bị và phát lệnh (launch kernel) lên GPU lớn hơn thời gian GPU tính toán thực tế.
    *   **CUDA Graphs**: Kỹ thuật ghi lại (capture) chuỗi các lệnh GPU vào một đồ thị cố định tĩnh. CPU chỉ cần kích hoạt đồ thị này một lần duy nhất, loại bỏ hoàn toàn launch overhead. Tuy nhiên, CUDA Graphs yêu cầu tiền cấp phát một lượng lớn VRAM cố định cho các hình dáng đầu vào (shapes) được bucketing trước.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Luôn để mặc định (tức là tắt `--enforce-eager`, cho phép chạy CUDA Graphs) để đạt hiệu năng latency tốt nhất trong các tác vụ thông thường.
    *   Chỉ bật `--enforce-eager` khi:
        1.  Bạn bị giới hạn cực kỳ nghiêm ngặt về VRAM và không thể dành ra $1 - 2$ GB VRAM đệm cho CUDA Graph capture.
        2.  Mô hình bạn chạy có kiến trúc động phức tạp hoặc sử dụng các thư viện custom không tương thích với cơ chế capture của CUDA Graphs.

### B. `--cuda-graph-max-seq-len <int>`
*   **Ý nghĩa**: Chiều dài ngữ cảnh tối đa mà đồ thị CUDA Graphs thực hiện ghi và tối ưu hóa (mặc định thường là $2048$ hoặc $4096$).
*   **Kiến thức nền tảng**: Phép chia nhóm hình dáng (Shape Bucketing) để tránh chụp lại đồ thị cho từng độ dài sequence riêng lẻ gây lãng phí bộ nhớ GPU.
*   **Kinh nghiệm thực chiến (Tuning Tricks)**:
    *   Nếu ứng dụng của bạn phục vụ các chuỗi rất dài (ví dụ hội thoại dài 8K hoặc 16K), hãy chú ý rằng việc capture CUDA Graphs cho sequence len lớn yêu cầu lượng VRAM khổng lồ. 
    *   Hãy giữ `--cuda-graph-max-seq-len` ở mức vừa phải ($4096$). Với các token vượt quá giới hạn này, vLLM sẽ tự động chuyển về chạy chế độ Eager Mode để bảo toàn VRAM.

---

## 💡 Tổng kết bài học và Bảng tra cứu nhanh

Dưới đây là bảng tổng hợp mối quan hệ giữa các tham số CLI cốt lõi và yêu cầu kỹ năng của kỹ sư:

| Tham số CLI | Mục tiêu tối ưu | Kiến thức nền tảng yêu cầu | Chỉ số giám sát cốt lõi |
| :--- | :--- | :--- | :--- |
| `--dtype` | Tính chính xác & VRAM | Định dạng số thực (BF16, FP16), Phần cứng GPU | Lỗi sinh số rác (NaN), Tốc độ hội tụ |
| `--tensor-parallel-size` | Băng thông đa GPU | Megatron-LM, Truyền thông NCCL, NVLink vs PCIe | Tốc độ truyền tải chéo GPU, Latency All-Reduce |
| `--gpu-memory-utilization` | Tránh GPU OOM | Cấu trúc bộ nhớ GPU, Activation Memory | VRAM Usage (Prometheus Metrics) |
| `--max-num-batched-tokens` | Saturation GPU | Mô hình Roofline, GEMM vs GEMV, Arithmetic Intensity | GPU Compute Utilization (%) |
| `--enable-chunked-prefill` | Giảm giật lag | Sự bất đối xứng của Prefill/Decode, Phép toán GEMM | Inter-Token Latency (ITL) |
| `--enable-prefix-caching` | Tăng tốc RAG/Chat | Cấu trúc dữ liệu Radix Tree, Cơ chế LRU Eviction | Prefix Cache Hit Rate (%) |
| `--enforce-eager` | Tiết kiệm VRAM | CUDA Graphs, CPU Launch Overhead, Shape Bucketing | Thời gian khởi động Engine, Độ hao phí VRAM |
