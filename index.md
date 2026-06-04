# vLLM Internals: Deep Dive into LLM Serving Architecture

Chào mừng bạn đến với tài liệu phân tích kiến trúc thư viện vLLM. Đây là tài liệu nghiên cứu chuyên sâu về các giải pháp tối ưu hóa hiệu năng, bộ lập lịch và quản lý bộ nhớ cho hệ thống LLM Serving.

---

## 🗺️ Lộ trình & Các Bài học (Syllabus & Navigation)

Vui lòng bấm vào liên kết dưới đây để truy cập trực tiếp nội dung chi tiết của từng phần:

* **🗺️ Lộ trình tổng quan**: [Chi tiết Giáo trình học tập & Lộ trình](README_ARCHIVED.html)

### 📚 Danh sách các bài giảng chi tiết:
0. **Bài 0: Kiến thức Hệ điều hành bổ trợ (OS Fundamentals for AI Serving)**  
   *Hiểu cơ chế phân trang bộ nhớ (Paging, Page Table), và các giao thức giao tiếp đa tiến trình (ZMQ, IPC, Shared Memory).*  
   👉 [Đọc Bài 0](docs/lesson_0_os_fundamentals.html)
   
1. **Bài 1: Autoregressive Serving & Memory Bottlenecks**  
   *Hiểu bản chất pha Prefill vs Decode, Arithmetic Intensity, công thức tính KV Cache và vấn đề phân mảnh.*  
   👉 [Đọc Bài 1](docs/lesson_1_memory_bottleneck.html)
   
2. **Bài 2: PagedAttention & Block Allocation**  
   *Nguyên lý phân trang bộ nhớ ảo áp dụng cho GPU, Logical vs Physical Blocks và Copy-on-Write.*  
   👉 [Đọc Bài 2](docs/lesson_2_paged_attention.html)

3. **Bài 3: Continuous Batching & Preemption**  
   *Lập lịch mức Iteration, chiến lược Preemption (Swap vs Recompute) và kỹ thuật Chunked Prefill.*  
   👉 [Đọc Bài 3](docs/lesson_3_continuous_batching.html)

4. **Bài 4: Async Serving, Concurrency & Streaming**  
   *Lập trình async, kiến trúc đa tiến trình Decoupled Engine qua ZMQ, stream token (SSE) và Abort.*  
   👉 [Đọc Bài 4](docs/lesson_4_async_concurrency.html)

5. **Bài 5: Codebase Deep Dive: Scheduler & Block Manager**  
   *Phân tích chi tiết code của Scheduler, KVCacheManager và Request Queue trong vLLM v1.*  
   👉 [Đọc Bài 5](docs/lesson_5_scheduler_code.html)

6. **Bài 6: Codebase Deep Dive: Distributed Executor & Workers**  
   *Song song hóa (TP/PP), Worker lifecycle, Memory Profiling và CUDA Graphs.*  
   👉 [Đọc Bài 6](docs/lesson_6_distributed_worker.html)

7. **Bài 7: Kỹ thuật Tối ưu hóa Nâng cao**  
   *Speculative Decoding, Multi-LoRA (Punica/SGMV kernels) và lượng hóa FP8/AWQ/Marlin.*  
   👉 [Đọc Bài 7](docs/lesson_7_advanced_serving.html)

8. **Bài 8: Thiết kế & Hiện thực Toy Serving Engine**  
   *Hướng dẫn tự viết một Serving Engine hoàn chỉnh hỗ trợ Paged Allocation, Continuous Batching và FastAPI Concurrent Streaming.*  
   👉 [Đọc Bài 8](docs/lesson_8_toy_serving_engine.html)

---

## 🛠️ Trải nghiệm cục bộ
Mã nguồn mô phỏng của **Bài 8** nằm trong thư mục `src/`. Bạn có thể clone dự án về máy và khởi chạy máy chủ theo hướng dẫn tại cuối tài liệu Bài 8 hoặc lộ trình tổng quan.
