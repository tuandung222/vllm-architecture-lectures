---
sidebar_position: 0
sidebar_label: "🗺️ Roadmap & Syllabus"
---

# vLLM Internals: Deep Dive into LLM Serving Architecture

Chào mừng bạn đến với kho lưu trữ bài giảng **vLLM Internals: Deep Dive into LLM Serving Architecture**. Chuỗi bài giảng này được thiết kế đặc biệt cho các **AI Serving Engineer, Deep Learning Engineer và Research Engineer** muốn tìm hiểu sâu sắc về cách thức hoạt động bên trong của vLLM - thư viện phục vụ mô hình ngôn ngữ lớn (LLM Serving) phổ biến và tối ưu nhất hiện nay.

Mục tiêu tối thượng của chuỗi bài giảng này là giúp bạn không chỉ **hiểu chi tiết từng dòng code** của vLLM, mà còn **nắm vững các nguyên lý thiết kế hệ thống suy luận** để trong tương lai có thể tự xây dựng một hệ thống AI Serving tối ưu của riêng mình.

---

## 🗺️ Lộ trình Chuỗi Bài Giảng (Roadmap)

Dự án được chia làm 9 bài học chính đi kèm mã nguồn mô phỏng chi tiết:

| Bài học | Chủ đề | Nội dung cốt lõi | Tài liệu |
| :--- | :--- | :--- | :--- |
| **Bài 1** | **Autoregressive Serving & Memory Bottleneck** | Phân tích pha Prefill vs Decode, Arithmetic Intensity, công thức tính KV Cache dung lượng lớn, phân mảnh VRAM. | [Bài 1 Docs](./02_single_node_memory_scheduling/lesson_1_memory_bottleneck.md), [Bài 1.1 Docs](./02_single_node_memory_scheduling/lesson_1_1_batch_size_compute_bound.md) |
| **Bài 2** | **PagedAttention & Block Allocation** | Nguyên lý Virtual Memory & Paging áp dụng vào GPU, Logical vs Physical Blocks, thuật toán PagedAttention Triton/CUDA Kernels, CoW (Copy-on-Write). | [Bài 2 Docs](./02_single_node_memory_scheduling/lesson_2_paged_attention.md), [Bài 2.2 Docs](./02_single_node_memory_scheduling/lesson_2_2_prefix_caching.md) |
| **Bài 3** | **Continuous Batching & Preemption** | Iteration-level scheduling, chiến lược thu hồi bộ nhớ (Recomputation vs Swapping), Chunked Prefill & Multi-step Execution tránh Latency Spike. | [Bài 3 Docs](./02_single_node_memory_scheduling/lesson_3_continuous_batching.md), [Bài 3.2 Docs](./02_single_node_memory_scheduling/lesson_3_2_chunked_prefill.md) |
| **Bài 4** | **Async Serving, Concurrency & Streaming** | Đa nhiệm bất đồng bộ (`asyncio`), Kiến trúc đa tiến trình Decoupled Engine (v1) qua ZeroMQ IPC/Shared Memory, xử lý truyền phát token (Streaming SSE), Abort Request khi client hủy kết nối. | [Bài 4 Docs](./03_decoupled_engine_async/lesson_4_async_concurrency.md), [Bài 4.1 Docs](./03_decoupled_engine_async/lesson_4_1_shared_memory_ipc.md) |
| **Bài 5** | **Deep Dive Codebase: Scheduler & Block Manager** | Khảo sát chi tiết mã nguồn v1 Scheduler (`scheduler.py`), quản lý hàng đợi (`request_queue.py`), quản lý khối KV Cache (`kv_cache_manager.py`). | [Bài 5 Docs](./04_vllm_core_codebase/lesson_5_scheduler_code.md) |
| **Bài 6** | **Deep Dive Codebase: Executor & GPU Workers** | Mô hình phân tán đa GPU (NCCL, Ray, Multiprocessing), quy trình khởi tạo ModelRunner & Memory Profiling, cơ chế CUDA Graphs Capture giảm CPU overhead. | [Bài 6 Docs](./05_distributed_multi_gpu/lesson_6_distributed_worker.md), [Bài 6.2 Docs](./05_distributed_multi_gpu/lesson_6_2_distributed_comm_nccl.md), [Bài 6.3 Docs](./05_distributed_multi_gpu/lesson_6_3_cuda_graph_bucketing.md), [Bài 6.4 Docs](./05_distributed_multi_gpu/lesson_6_4_ray_multiprocessing_orchestration.md), [Bài 6.5 Docs](./05_distributed_multi_gpu/lesson_6_5_moe_expert_parallelism_eplb.md), [Bài 6.6 Docs](./05_distributed_multi_gpu/lesson_6_6_nccl_bottlenecks_networking.md), [Bài 6.7 Docs](./05_distributed_multi_gpu/lesson_6_7_context_parallelism_ring_attention.md), [Bài 6.8 Docs](./05_distributed_multi_gpu/lesson_6_8_data_parallelism_replicas_routing.md) |
| **Bài 7** | **Tối ưu hóa Nâng cao cho AI Serving** | Speculative Decoding (Verify nháp song song), Multi-LoRA Serving (Punica/SGMV kernels), Lượng hóa nâng cao (FP8, INT8, AWQ, Marlin), Multimodal serving (Vision/Audio nạp dữ liệu và bộ đệm). | [Bài 7 Docs](./07_speculative_decoding_disaggregation/lesson_7_advanced_serving.md), [Bài 7.1 Docs](./06_vram_optimization/lesson_7_quantization_deep_dive.md), [Bài 7.2 Docs](./07_speculative_decoding_disaggregation/lesson_7_2_speculative_decoding_deep_dive.md), [Bài 7.2.1 Docs](./07_speculative_decoding_disaggregation/lesson_7_2_1_speculative_decoding_vllm_impl.md), [Bài 7.2.2 Docs](./07_speculative_decoding_disaggregation/lesson_7_2_2_speculative_decoding_production.md), [Bài 7.3 Docs](./06_vram_optimization/lesson_7_3_multi_lora_serving.md), [Bài 7.4 Docs](./07_speculative_decoding_disaggregation/lesson_7_4_prefill_decode_disaggregation.md), [Bài 7.5 Docs](./08_multimodal_serving/lesson_7_5_multimodal_architecture_vllm.md), [Bài 7.6 Docs](./08_multimodal_serving/lesson_7_6_multimodal_cache_memory.md), [Bài 7.7 Docs](./08_multimodal_serving/lesson_7_7_multimodal_production_gotchas.md) |
| **Bài 8** | **Thiết kế & Thực hành: Toy Serving Engine** | Tổng kết kiến trúc tự build và hiện thực thực tế một Serving Engine có Paged Block Allocator, Continuous Batching, Async FastAPI Server hỗ trợ Concurrent Streaming & Abort. | [Bài 8 Docs](./09_toy_serving_engine/lesson_8_toy_serving_engine.md) |
| **Bài 9** | **Cẩm nang Tinh chỉnh CLI & Tham chiếu Production** | Tra cứu chi tiết toàn bộ các tham số CLI quan trọng trong vLLM, các kiến thức nền tảng và mẹo tuning thực chiến để kiểm soát VRAM, tránh OOM. | [Bài 9 Docs](./10_production_reference/lesson_9_vllm_cli_reference.md) |
| **Bài 10** | **Quản lý VRAM & Chiến lược Lập lịch Offline Batching** | Cơ chế phân bổ VRAM tĩnh, profiling run xác định activation memory, giới hạn Scheduler và phân tích sâu 4 tính năng tối ưu: Prefix Caching, Chunked Prefill, Swapping, CUDA Graphs. | [Bài 10 Docs](./10_production_reference/lesson_10_vram_allocation_offline_batching.md) |

---

## 📂 Cấu trúc Repository

```bash
vllm-architecture-lectures/
├── README.md                          # Giới thiệu tổng quan lộ trình học
├── docs/                              # Tài liệu biên soạn theo 10 Modules chuyên đề
│   ├── 01_system_foundations/         # Module 1: Nền tảng Phần cứng GPU & Hệ điều hành
│   ├── 02_single_node_memory_scheduling/ # Module 2: Quản lý Bộ nhớ & Lập lịch Đơn Node
│   ├── 03_decoupled_engine_async/     # Module 3: Kiến trúc Engine Tách rời & Xử lý Bất đồng bộ
│   ├── 04_vllm_core_codebase/         # Module 4: Phân tích Mã nguồn vLLM Core
│   ├── 05_distributed_multi_gpu/      # Module 5: Kỹ thuật Phục vụ Phân tán & Đa GPU
│   ├── 06_vram_optimization/          # Module 6: Tối ưu hóa VRAM: Lượng hóa & Phục vụ Đa LoRA
│   ├── 07_speculative_decoding_disaggregation/ # Module 7: Giải mã Suy đoán & Kiến trúc Phân rã
│   ├── 08_multimodal_serving/         # Module 8: Phục vụ Mô hình Đa phương thức (LMM/VLM Serving)
│   ├── 09_toy_serving_engine/         # Module 9: Thực hành: Hiện thực hóa Serving Engine tối giản
│   ├── 10_production_reference/       # Module 10: Tài liệu Tham chiếu Production (CLI Reference)
│   └── roadmap.md                     # Giáo trình và mục lục tổng thể của kho khóa học
└── toy_engine/                        # Mã nguồn thực hành tự build Toy Serving Engine (Bài 8)
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

Bạn nên bắt đầu đọc từ **[Bài 1: Autoregressive Serving & Memory Bottleneck](./02_single_node_memory_scheduling/lesson_1_memory_bottleneck.md)** để nắm rõ bài toán cốt lõi mà vLLM đang giải quyết, sau đó đi tuần tự theo lộ trình học tập. 

Tại **Bài 8**, chúng ta sẽ cùng nhau chạy thử mã nguồn tại thư mục `/src` để kiểm nghiệm lại toàn bộ lý thuyết đã học!
