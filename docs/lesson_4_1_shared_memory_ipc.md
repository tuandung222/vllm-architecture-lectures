---
sidebar_position: 7.5
sidebar_label: "Bài 4.1: SHM & ZeroMQ IPC trong Decoupled Engine"
---

# Bài 4.1: Đi sâu Cơ chế Shared Memory & ZeroMQ IPC trong Decoupled Engine (v1)

Trong [Bài 4](./lesson_4_async_concurrency.md), chúng ta đã làm quen với kiến trúc **Decoupled Engine (vLLM v1)**. Đây là thiết kế tách biệt hoàn toàn tiến trình xử lý API HTTP (Frontend API Process) khỏi tiến trình điều phối tính toán GPU (Backend Engine Process).

Tuy nhiên, sự tách biệt này mở ra một thách thức kỹ thuật cực kỳ lớn về **Giao tiếp giữa các Tiến trình (Inter-Process Communication - IPC)**: Làm thế nào để truyền nhận thông điệp điều khiển có độ trễ cực thấp (micro-seconds) và chia sẻ các Tensor dữ liệu dung lượng hàng trăm Megabytes giữa các tiến trình mà không làm nghẽn CPU và nghẽn băng thông hệ thống?

Bài học này sẽ mổ xẻ cơ chế giải quyết của vLLM thông qua sự phối hợp của **ZeroMQ** và **Shared Memory (SHM)**.

---

## 1. Nút thắt cổ chai: Python GIL và Overhead Sao chép Bộ nhớ (Memory Copying)

Trong các máy chủ phục vụ mô hình thông thường, API Server và GPU Engine thường chạy chung trên một tiến trình Python. Cách làm này gặp phải hai giới hạn nghiêm trọng:
1.  **GIL (Global Interpreter Lock)**: Python chỉ cho phép một luồng chạy tại một thời điểm. Khi GPU Engine bận chuẩn bị dữ liệu và launch CUDA kernels, luồng xử lý API FastAPI sẽ bị chặn (block), dẫn đến việc không thể nhận hoặc stream token cho các request khác kịp thời.
2.  **IPC Overhead**: Nếu tách thành hai tiến trình độc lập, việc truyền Tensor qua các cơ chế IPC tiêu chuẩn (như gRPC, Unix Domain Sockets, hoặc Multi-processing Pipes) yêu cầu dữ liệu phải trải qua quá trình **Tuần tự hóa (Serialization/Pickling)** ở tiến trình gửi, ghi qua kernel socket, và giải tuần tự hóa ở tiến trình nhận. Quá trình này ngốn rất nhiều tài nguyên CPU và tạo độ trễ lớn (Latency Bottleneck).

```
MÔ HÌNH IPC THƯỜNG (CHẬM):
[API Server] ➔ (Serialize) ➔ (Copy vào Kernel) ➔ (Unix Socket) ➔ (Copy từ Kernel) ➔ (Deserialize) ➔ [GPU Worker]
```

Để tối ưu hóa, vLLM v1 phân tách nhiệm vụ giao tiếp thành hai kênh chuyên biệt: **Kênh điều khiển (Control Plane)** sử dụng ZeroMQ, và **Kênh dữ liệu (Data Plane)** sử dụng Shared Memory.

---

## 2. Kênh Điều Khiển (Control Plane): ZeroMQ IPC Sockets

Kênh điều khiển chịu trách nhiệm truyền tải các thông điệp có dung lượng nhỏ nhưng tần suất cực cao: metadata của request mới, lệnh hủy (abort), danh sách token sinh ra ở mỗi bước.

vLLM sử dụng **ZeroMQ (ZMQ)** chạy trên giao thức IPC (`ipc://` socket file cục bộ trên OS):
* **Không chặn (Non-blocking I/O)**: ZMQ tự động quản lý các hàng đợi gửi/nhận ở tầng C++ chạy ngầm. FastAPI gửi chỉ thị cho EngineCore mà không bị block đợi phản hồi.
* **Mô hình Router - Dealer**:
  * **ZMQ ROUTER** (ở phía API Server): Nhận hàng ngàn yêu cầu từ các client đồng thời, tự động gán nhãn nhận diện (Identity Envelope) cho từng request.
  * **ZMQ DEALER** (ở phía GPU EngineCore): Nhận luồng thông điệp không trạng thái, đưa vào hàng đợi `RequestQueue`. Khi GPU sinh xong token, Dealer gửi trả ngược lại cho Router kèm nhãn nhận diện để chuyển đúng luồng SSE (Server-Sent Events) về cho client tương ứng.

Vì kích thước của token ID hay metadata chỉ cỡ vài Bytes, chi phí serialize bằng **Msgpack** (thư viện serialize nhị phân siêu nhanh, thay thế cho JSON) qua ZMQ là không đáng kể.

---

## 3. Kênh Dữ Liệu (Data Plane): Shared Memory (SHM) - Zero-Copy Tensor Transfer

Đối với các dữ liệu kích thước lớn (như tensor đầu vào của prompt, ảnh/video trong mô hình đa phương thức Multimodal, hoặc KV Cache trung gian), vLLM hoàn toàn không truyền qua ZMQ. Thay vào đó, vLLM sử dụng cơ chế **Shared Memory (Bộ nhớ chia sẻ)**.

```
MÔ HÌNH SHM TRONG vLLM (ZERO-COPY):
      [API Server Process]                [GPU Worker Process]
    Không gian địa chỉ ảo A              Không gian địa chỉ ảo B
   ┌──────────────────────┐             ┌──────────────────────┐
   │  [Vùng nhớ ảo A]     │             │  [Vùng nhớ ảo B]     │
   └──────────┬───────────┘             └───────────┬──────────┘
              │                                     │
              +-------------> [ RAM Vật lý ] <------+
                          (Shared Memory: /dev/shm)
```

### 3.1. Nguyên lý hoạt động của SHM:
1.  **Ánh xạ vùng nhớ chung (Memory Mapping)**: Khi khởi chạy, vLLM tạo ra một vùng nhớ chia sẻ trên RAM hệ thống (thường sử dụng file ảo trong hệ thống tệp `/dev/shm` trên Linux). Cả tiến trình API Server và tiến trình GPU Worker đều thực hiện ánh xạ (mmap) vùng nhớ vật lý này vào không gian nhớ ảo của riêng mình.
2.  **Zero-Copy Write**: Khi nhận được Prompt, tiến trình API Server mã hóa văn bản thành danh sách token ID và viết trực tiếp vào vùng nhớ SHM này.
3.  **Zero-Copy Read**: Tiến trình GPU Worker nhận được thông điệp ZMQ báo hiệu có dữ liệu mới, nó chỉ việc đọc trực tiếp dữ liệu từ vùng nhớ SHM của mình. **Không có bất kỳ hành vi sao chép vật lý nào xảy ra trên RAM** (Zero-copy). 

### 3.2. Quản lý vùng SHM (Lock-free Ring Buffer):
Để tránh xung đột ghi/đọc giữa hai tiến trình mà không dùng các cơ chế khóa (locks, mutexes) làm chậm hệ thống, vLLM sử dụng cấu trúc dữ liệu **Lock-free Ring Buffer (Vòng đệm không khóa)** kết hợp với các biến đếm nguyên tử (Atomic Counters). API Server ghi dữ liệu vào một đầu của buffer, GPU Worker đọc dữ liệu từ đầu kia dựa trên con trỏ chỉ số (Index Pointer).

---

## 4. Minh họa luồng đi của Dữ liệu trong Codebase vLLM v1

*   **Đăng ký Request**: Khi client gửi request tới [vllm/entrypoints/openai/api_server.py](file:///Users/admin/TuanDung/repos/vllm/vllm/entrypoints/openai/api_server.py), API Server đóng gói metadata của request bằng Msgpack và đẩy qua ZMQ. Đồng thời, ghi mảng Tensor tokens đầu vào vào bộ nhớ SHM.
*   **Xử lý ở GPU**: Tệp [vllm/v1/engine/core.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/v1/engine/core.py) (chạy tiến trình GPU độc lập) lắng nghe ZMQ, đọc trực tiếp dữ liệu tokens đầu vào từ vùng nhớ SHM chung, nạp vào GPU và thực hiện chuỗi tính toán.
*   **Trả kết quả**: GPU sinh ra Token ID mới, đóng gói qua ZMQ gửi ngược lại cho API Server để stream về cho client.

---

## 💡 Tổng kết bài học

* **Decoupled Engine** trong vLLM v1 giải quyết triệt để vấn đề nghẽn Python GIL bằng cách tách biệt hoàn toàn Frontend API và Backend GPU.
* **ZMQ IPC** đóng vai trò là **Control Plane** truyền tải siêu nhanh các gói tin điều khiển kích thước nhỏ qua cơ chế không chặn (non-blocking Router-Dealer).
* **Shared Memory (SHM)** đóng vai trò là **Data Plane** thực hiện truyền tải **Zero-copy** đối với các Tensor đầu vào kích thước lớn, loại bỏ hoàn toàn CPU/Memory overhead của việc sao chép dữ liệu giữa các tiến trình.
* Sự kết hợp của hai công nghệ này giúp vLLM v1 đạt được hiệu suất phục vụ cực đại với độ trễ giao tiếp ở mức micro-seconds, sẵn sàng cho các luồng dữ liệu đa phương thức (Multimodal) khổng lồ.
