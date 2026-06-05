---
sidebar_position: 11.85
sidebar_label: "Bài 7.6: Quản lý bộ đệm & VRAM trong Multimodal serving"
---

# Bài 7.6: Quản lý bộ đệm và tối ưu hóa VRAM cho Multimodal Serving

Việc xử lý các dữ liệu hình ảnh, video dung lượng lớn trên GPU tạo ra áp lực cực lớn lên băng thông truyền tin và bộ nhớ VRAM. Nếu không có cơ chế quản lý bộ đệm thông minh, hệ thống sẽ liên tục phải tính toán lại các đặc trưng ảnh đắt đỏ hoặc bị sập tiến trình do tràn bộ nhớ (OOM) khi xử lý nhiều hình ảnh đồng thời.

Bài học này sẽ mổ xẻ cơ chế hoạt động của bộ đệm **Shared Memory Zero-Copy** và giải thuật tính toán ngân sách bộ nhớ **Encoder Budget** bên trong vLLM.

---

## 1. Tại sao chúng ta cần bộ đệm đặc trưng đa phương thức (Multimodal Cache)?

Khi người dùng thực hiện một cuộc hội thoại đa lượt (multi-turn conversation) với mô hình đa phương thức:
*   *Lượt 1*: Người dùng gửi 1 bức ảnh kèm câu hỏi *"Bức ảnh này vẽ gì?"*. Hệ thống chạy Vision Tower để trích xuất đặc trưng ảnh thô, chiếu thành visual tokens và đưa vào LLM để sinh câu trả lời.
*   *Lượt 2*: Người dùng hỏi tiếp *"Hãy chỉ ra chi tiết A nằm ở góc nào của bức ảnh đó"*.

Nếu không có cơ chế lưu trữ bộ đệm đặc trưng:
1.  Hệ thống bắt buộc phải nạp lại bức ảnh thô từ người dùng.
2.  Khởi chạy lại toàn bộ forward pass của Vision Tower trên GPU để tạo lại visual tokens từ đầu.

Do bộ mã hóa Vision Tower (như ViT hay SigLIP) chạy tính toán GEMM rất nặng, việc lặp lại tính toán này cho cùng một bức ảnh qua nhiều lượt chat gây lãng phí năng lực tính toán của GPU và làm tăng vọt độ trễ phản hồi của token đầu tiên (Time-to-First-Token - TTFT). Bộ đệm **Multimodal Processor Cache** ra đời để lưu trữ lại các đặc trưng ảnh đã xử lý xong và tái sử dụng tức thì.

---

## 2. Cơ chế đệm Shared Memory Zero-Copy (P0 vs P1 Cache)

Trong kiến trúc tách biệt tiến trình của vLLM v1:
*   **Tiến trình API (P0)**: Nhận ảnh thô từ client, thực hiện xử lý bằng Python processor.
*   **Tiến trình GPU Worker (P1)**: Chạy mô hình trên GPU.

Để chuyển các tensor đặc trưng hình ảnh (thường có kích thước rất lớn, hàng chục MB) từ P0 sang P1 mà không tốn chi phí copy dữ liệu qua mạng hoặc IPC truyền thống, vLLM v1 hiện thực hóa cơ chế đệm Shared Memory thông qua [cache.py](file:///Users/admin/TuanDung/repos/vllm/vllm/multimodal/cache.py):

```
 [ Tiến trình API (P0) ]                        [ Tiến trình GPU Worker (P1) ]
         │                                                    │
         ├────── (Put Tensor đặc trưng ảnh)                    │
         ▼                                                    │
┌──────────────────────────────────────┐                      │
│     SingleWriterShmRingBuffer        │                      │
│           (Shared Memory)            │                      │
└──────────────────────────────────────┘                      │
         ▲                                                    │
         └────── (Gửi address & monotonic_id qua ZMQ) ───────>│
                                                              ├────── (Đọc trực tiếp - Zero Copy)
                                                              ▼
```

### 2.1. Phân tích chi tiết luồng Zero-Copy IPC qua Shared Memory

Trong các hệ thống phân tán thông thường, việc truyền các tensor lớn (ví dụ ảnh pixel thô hoặc đặc trưng ảnh) giữa hai tiến trình khác nhau thường sử dụng TCP Socket hoặc Unix Domain Socket. Tuy nhiên, phương pháp này đòi hỏi:
1.  **Tuần tự hóa (Serialization)** tại P0.
2.  **Sao chép bộ nhớ (Memory Copy)** từ không gian người dùng (User space) của P0 sang không gian nhân hệ điều hành (Kernel space), sau đó từ Kernel space sang User space của P1.
3.  **Giải tuần tự hóa (Deserialization)** tại P1.

Đối với dữ liệu ảnh độ phân giải cao hoặc chuỗi khung hình video, chi phí sao chép bộ nhớ chéo tiến trình (IPC Overhead) này cực kỳ lớn, gây thắt nút cổ chai nghiêm trọng tại CPU.

Để giải quyết triệt để, vLLM áp dụng giải pháp **Zero-Copy Shared Memory**:

*   **Vùng nhớ dùng chung (POSIX Shared Memory)**: vLLM ánh xạ cùng một vùng nhớ vật lý của hệ thống (RAM) vào không gian địa chỉ ảo của cả hai tiến trình P0 và P1.
*   **Cơ chế ghi đơn (SingleWriterShmRingBuffer)**: P0 đóng vai trò là Writer duy nhất ghi tuần tự vào bộ đệm vòng (Ring Buffer) dùng chung này. Khi nhận ảnh mới, P0 ghi trực tiếp dữ liệu thô vào buffer.
*   **Truyền tin nhắn ZMQ siêu nhẹ**: Thay vì truyền toàn bộ dữ liệu Tensor qua ZeroMQ socket, P0 chỉ gửi một gói tin JSON/Msgpack nhỏ chứa metadata bao gồm:
    *   Địa chỉ con trỏ của khối bộ nhớ (`address`).
    *   Kích thước và hình dạng của tensor (`shape`, `dtype`).
    *   Mã định danh duy nhất tăng dần (`monotonic_id`).
*   **Khởi tạo Tensor Zero-Copy**: Tại tiến trình P1, `ShmObjectStoreReceiverCache` nhận thông tin địa chỉ. Nó sử dụng thư viện thích hợp (như NumPy và PyTorch) để map trực tiếp con trỏ vùng nhớ Shared Memory đó thành một CPU Tensor thông qua `torch.from_numpy()` mà không tốn bất kỳ chu kỳ CPU nào cho việc copy dữ liệu (Zero-copy). Sau đó, tensor này mới được chuyển (transfer) lên GPU để đưa vào mô hình.

### 2.2. Cơ chế đồng bộ hóa bộ đệm chéo tiến trình:

1.  **Ghi vào Shared Memory**: Khi P0 tiền xử lý hoặc nhận diện ảnh mới, nó gọi `ShmObjectStoreSenderCache.get_and_update_item()`. Nếu cache miss, P0 sẽ ghi trực tiếp tensor đặc trưng ảnh vào bộ đệm vòng dùng chung của hệ điều hành `SingleWriterShmRingBuffer`.
2.  **Gửi con trỏ địa chỉ**: Thay vì truyền toàn bộ tensor đặc trưng qua ZeroMQ IPC, P0 chỉ gửi tin nhắn nhỏ chứa địa chỉ vùng nhớ vật lý (`address`) và mã định danh (`monotonic_id`).
3.  **Đọc trực tiếp (Zero-Copy)**: Tại GPU Worker (P1), lớp `ShmObjectStoreReceiverCache` nhận thông tin địa chỉ và trỏ trực tiếp đến vùng nhớ Shared Memory đó để nạp tensor vào Vision Tower của mô hình.
4.  **Chính sách trục xuất (LRU Eviction)**: Cả P0 và P1 duy trì các bản sao cache metadata đồng bộ. Khi bộ đệm đầy, thuật toán LRU sẽ tự động giải phóng vùng nhớ Shared Memory ở cả hai phía đồng bộ mà không cần truyền thông tin phối hợp chéo.

---

## 3. Lập ngân sách bộ nhớ: MultiModalBudget

Việc xử lý ảnh/video trên batch yêu cầu một lượng tài nguyên VRAM rất lớn cho các bộ đệm trung gian của Vision Tower. Để ngăn ngừa lỗi tràn bộ nhớ GPU (OOM) khi batch size tăng cao, vLLM quản lý thông qua lớp `MultiModalBudget` trong [encoder_budget.py](file:///Users/admin/TuanDung/repos/vllm/vllm/multimodal/encoder_budget.py).

`MultiModalBudget` thực hiện tính toán giới hạn an toàn trước khi nạp batch:

### A. Tính toán ngân sách tối đa
Hệ thống xác định giới hạn thông qua hàm `compute_mm_encoder_budget()` dựa trên cấu hình bộ lập lịch:
*   `encoder_compute_budget`: Giới hạn lượng tính toán tối đa cho bộ mã hóa.
*   `encoder_cache_size`: Giới hạn dung lượng lưu trữ tối đa cho đặc trưng ảnh trong cache.

### B. Giới hạn số lượng ảnh trên Prompt và Batch
Thông qua phương thức `_get_max_items()`, vLLM tính toán chính xác số lượng ảnh tối đa được phép xử lý:
1.  **Trên mỗi Prompt (`mm_max_items_per_prompt`)**: Được giới hạn bởi độ dài tối đa của mô hình và dung lượng KV Cache.
2.  **Trên mỗi Batch (`mm_max_items_per_batch`)**: Được tính toán động dựa trên lượng bộ đệm còn trống của encoder và decoder:

$$\text{max\_items\_per\_batch} = \min\left(\frac{\text{encoder\_budget}}{\text{max\_tokens\_per\_item}}, \text{max\_num\_reqs} \times \text{max\_items\_per\_prompt}\right)$$

Nếu một batch vượt quá giới hạn này, bộ lập lịch (Scheduler) sẽ tự động trì hoãn (preempt) các request đa phương tiện tiếp theo sang lượt lặp (iteration) sau để bảo vệ an toàn cho VRAM GPU.

---

## 4. Liên hệ với Toy Engine: Sự đơn giản hóa của KV Cache tĩnh

Hãy đối chiếu cơ chế quản lý động này với mô phỏng [Toy Serving Engine](file:///Users/admin/TuanDung/vllm-architecture-lectures/toy_engine/) của chúng ta:

*   **Toy Engine (Bộ lập lịch tĩnh - Static Scheduler)**: Trong [scheduler.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/toy_engine/scheduler.py) (`ToyScheduler`), bộ lập lịch hoạt động cực kỳ đơn giản: nó chỉ việc đếm số lượng khối KV Cache trống của mô hình ngôn ngữ chính để quyết định có nhận thêm request mới hay không. Bộ lập lịch giả định không gian bộ nhớ của GPU hoàn toàn tĩnh và chỉ dùng cho LLM.
*   **Production vLLM (Bộ lập lịch động - Dynamic VLM Scheduler)**: Trong thực tế serving VLM, bộ lập lịch phải phối hợp đồng thời với **Block Manager** của LLM và **MultiModalBudget** của Vision Tower. Nó không chỉ kiểm tra dung lượng block KV Cache trống của decoder, mà còn phải liên tục kiểm tra không gian bộ đệm đặc trưng thị giác trống (`encoder_cache_size`) và khống chế nghiêm ngặt số lượng ảnh đang xử lý đồng thời trong batch (`mm_max_items_per_batch`). Sự phối hợp động này giúp GPU phân bổ tối ưu tài nguyên giữa pha trích xuất đặc trưng ảnh (Vision) và pha sinh từ tự hồi quy (Language).

---

## 💡 Tổng kết bài học

*   **Multimodal Cache** lưu trữ lại các đặc trưng ảnh đã xử lý xong để tránh chạy lại Vision Tower đắt đỏ trong hội thoại đa lượt, giúp giảm thiểu đáng kể Time-to-First-Token (TTFT).
*   vLLM v1 tối ưu hóa truyền tin chéo tiến trình bằng cơ chế **Zero-Copy Shared Memory** (`ShmObjectStoreSenderCache` / `ShmObjectStoreReceiverCache`), truyền tọa độ địa chỉ thay vì copy toàn bộ tensor ảnh lớn.
*   **MultiModalBudget** trong [encoder_budget.py](file:///Users/admin/TuanDung/repos/vllm/vllm/multimodal/encoder_budget.py) tính toán giới hạn an toàn vật lý của VRAM, tự động khống chế số lượng ảnh tối đa trên prompt (`mm_max_items_per_prompt`) và trên batch (`mm_max_items_per_batch`) để ngăn ngừa GPU OOM.
