# vLLM Internals: Deep Dive into LLM Serving Architecture

Chào mừng bạn đến với kho lưu trữ bài giảng **vLLM Internals: Deep Dive into LLM Serving Architecture**. Chuỗi bài giảng này được thiết kế đặc biệt cho các **AI Serving Engineer, Deep Learning Engineer và Research Engineer** muốn tìm hiểu sâu sắc về cách thức hoạt động bên trong của vLLM — thư viện phục vụ mô hình ngôn ngữ lớn (LLM Serving) phổ biến và tối ưu nhất hiện nay.

Mục tiêu tối thượng của chuỗi bài giảng này là giúp bạn không chỉ **hiểu chi tiết từng dòng code** của vLLM, mà còn **nắm vững các nguyên lý thiết kế hệ thống suy luận** để trong tương lai có thể tự xây dựng một hệ thống AI Serving tối ưu của riêng mình.

---

## 🗺️ Lộ trình Chuỗi Bài Giảng (Roadmap)

Dự án được chia làm 8 bài học chính đi kèm mã nguồn mô phỏng chi tiết:

| Bài học | Chủ đề | Nội dung cốt lõi | Tài liệu |
| :--- | :--- | :--- | :--- |
| **Bài 1** | **Autoregressive Serving & Memory Bottleneck** | Phân tích pha Prefill vs Decode, Arithmetic Intensity, công thức tính KV Cache dung lượng lớn, phân mảnh VRAM. | [Bài 1 Docs](docs/lesson_1_memory_bottleneck.md) |
| **Bài 2** | **PagedAttention & Block Allocation** | Nguyên lý Virtual Memory & Paging áp dụng vào GPU, Logical vs Physical Blocks, thuật toán PagedAttention Triton/CUDA Kernels, CoW (Copy-on-Write). | [Bài 2 Docs](docs/lesson_2_paged_attention.md) |
| **Bài 3** | **Continuous Batching & Preemption** | Iteration-level scheduling, chiến lược thu hồi bộ nhớ (Recomputation vs Swapping), Chunked Prefill & Multi-step Execution tránh Latency Spike. | [Bài 3 Docs](docs/lesson_3_continuous_batching.md) |
| **Bài 4** | **Async Serving, Concurrency & Streaming** | Đa nhiệm bất đồng bộ (`asyncio`), Kiến trúc đa tiến trình Decoupled Engine (v1) qua ZeroMQ IPC/Shared Memory, xử lý truyền phát token (Streaming SSE), Abort Request khi client hủy kết nối. | [Bài 4 Docs](docs/lesson_4_async_concurrency.md) |
| **Bài 5** | **Deep Dive Codebase: Scheduler & Block Manager** | Khảo sát chi tiết mã nguồn v1 Scheduler (`scheduler.py`), quản lý hàng đợi (`request_queue.py`), quản lý khối KV Cache (`kv_cache_manager.py`). | [Bài 5 Docs](docs/lesson_5_scheduler_code.md) |
| **Bài 6** | **Deep Dive Codebase: Executor & GPU Workers** | Mô hình phân tán đa GPU (NCCL, Ray, Multiprocessing), quy trình khởi tạo ModelRunner & Memory Profiling, cơ chế CUDA Graphs Capture giảm CPU overhead. | [Bài 6 Docs](docs/lesson_6_distributed_worker.md) |
| **Bài 7** | **Tối ưu hóa Nâng cao cho AI Serving** | Speculative Decoding (Verify nháp song song), Multi-LoRA Serving (Punica/SGMV kernels), Lượng hóa nâng cao (FP8, INT8, AWQ, Marlin). | [Bài 7 Docs](docs/lesson_7_advanced_serving.md) |
| **Bài 8** | **Thiết kế & Thực hành: Toy Serving Engine** | Tổng kết kiến trúc tự build và hiện thực thực tế một Serving Engine có Paged Block Allocator, Continuous Batching, Async FastAPI Server hỗ trợ Concurrent Streaming & Abort. | [Bài 8 Docs](docs/lesson_8_toy_serving_engine.md) |

---

## 📂 Cấu trúc Repository

```bash
vllm-architecture-lectures/
├── README.md                          # Giới thiệu tổng quan lộ trình học
├── docs/                              # Thư mục lưu trữ tài liệu 8 bài giảng lý thuyết & phân tích code
│   ├── lesson_1_memory_bottleneck.md
│   ├── lesson_2_paged_attention.md
│   ├── lesson_3_continuous_batching.md
│   ├── lesson_4_async_concurrency.md
│   ├── lesson_5_scheduler_code.md
│   ├── lesson_6_distributed_worker.md
│   ├── lesson_7_advanced_serving.md
│   └── lesson_8_toy_serving_engine.md
└── src/                               # Mã nguồn thực hành tự build Toy Serving Engine (Bài 8)
    ├── allocator.py                   # Quản lý Logical/Physical Blocks & Page Table
    ├── scheduler.py                   # Lập lịch Continuous Batching (Waiting, Running, Swapped)
    ├── model.py                       # Giả lập mô hình Transformer sinh token và đo latency
    ├── app.py                         # FastAPI Server chạy Engine loop bất đồng bộ & Streaming SSE
    └── client.py                      # Giả lập gửi đồng thời nhiều request để kiểm tra hệ thống
```

---

## 🛠️ Yêu cầu chuẩn bị (Prerequisites)

Để tiếp thu tốt nhất chuỗi bài giảng này, người học nên trang bị trước:
1. **Python nâng cao**: Hiểu lập trình bất đồng bộ (`asyncio`, `async/await`, `generator`).
2. **Deep Learning cơ bản**: Đã hiểu cấu trúc mô hình Transformer (Attention, MLP, LayerNorm) và cách tự hồi quy (autoregressive) sinh văn bản hoạt động.
3. **Kiến thức Hệ điều hành (OS) cơ bản**: Hiểu cơ chế phân trang bộ nhớ (Paging, Page Table), giao tiếp giữa các tiến trình (ZMQ, IPC, Shared Memory).
4. **PyTorch**: Đọc hiểu code tensor cơ bản. Không bắt buộc phải thành thạo CUDA/Triton vì chúng ta sẽ diễn giải phần kernel thông qua mã giả hoặc sơ đồ minh họa trực quan.

---

## 🚀 Bắt đầu như thế nào?

Bạn nên bắt đầu đọc từ **[Bài 1: Autoregressive Serving & Memory Bottleneck](docs/lesson_1_memory_bottleneck.md)** để nắm rõ bài toán cốt lõi mà vLLM đang giải quyết, sau đó đi tuần tự theo lộ trình học tập. 

Tại **Bài 8**, chúng ta sẽ cùng nhau chạy thử mã nguồn tại thư mục `/src` để kiểm nghiệm lại toàn bộ lý thuyết đã học!
