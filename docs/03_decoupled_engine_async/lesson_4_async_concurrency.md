---
sidebar_position: 6
sidebar_label: "Bài 4: Async Serving & Concurrency"
---

# Bài 4: Async Serving, Concurrency & Streaming (Kiến trúc Bất đồng bộ & Xử lý Đồng thời)

Một hệ thống AI serving trong môi trường sản xuất (production) phải đối mặt với áp lực xử lý hàng trăm hoặc hàng ngàn request đồng thời từ nhiều người dùng. Trong bài này, chúng ta sẽ khảo sát cách vLLM thiết kế hệ thống lập trình bất đồng bộ (`asyncio`) và kiến trúc tách biệt (**Decoupled Engine**) ở phiên bản vLLM v1 mới nhất để đạt được khả năng xử lý đồng thời cực cao mà không làm nghẽn tiến trình tính toán của GPU.

---

## 1. Thách thức GIL và Giải pháp Đa tiến trình (Decoupled Engine)

Trong ngôn ngữ Python, cơ chế **GIL (Global Interpreter Lock)** ngăn cản nhiều luồng (threads) thực thi mã Python cùng một lúc. 
* Nếu chúng ta chạy API Server (FastAPI), bộ lập lịch (Scheduler) và mã suy luận GPU (PyTorch) trên cùng một luồng hoặc tiến trình Python: Các tác vụ I/O của mạng (như nhận prompt, serialize kết quả JSON) sẽ trực tiếp làm nghẽn vòng lặp chạy mô hình của GPU. GPU sẽ phải dừng lại để đợi CPU thực hiện các công việc mạng.
* **Giải pháp của vLLM v1**: Tách biệt hoàn toàn hệ thống Serving thành 2 phân hệ độc lập chạy trên các tiến trình khác nhau:

```
+-----------------------------------------------------------+
|               Tiến trình 1: Frontend API Server           |
|  - FastAPI (HTTP/gRPC endpoints)                          |
|  - AsyncLLMEngine (Nhận & Detokenize kết quả)              |
|  - EngineCoreClient                                       |
+-----------------------------------------------------------+
                             |
                   ZeroMQ IPC Sockets (Bất đồng bộ)
                             |
+-----------------------------------------------------------+
|               Tiến trình 2: Backend Engine Core           |
|  - EngineCoreServer (Vòng lặp step() liên tục)            |
|  - Scheduler (Lập lịch phân trang bộ nhớ)                 |
|  - Executor / GPU Workers (Thực thi mô hình trên GPU)      |
+-----------------------------------------------------------+
```

### Cách giao tiếp đa tiến trình (Inter-process Communication):
* **ZeroMQ (ZMQ)**: Sử dụng các socket ZMQ tốc độ cao chạy trên cơ chế IPC (Inter-Process Communication) cục bộ hoặc TCP.
* **Msgpack Serialization**: Tránh việc serialize bằng JSON chậm chạp. vLLM sử dụng thư viện `msgspec` để mã hóa các thông điệp request/output thành định dạng nhị phân Msgpack siêu nhanh.
* **Shared Memory (IPC)**: Đối với các tensor dữ liệu lớn hoặc các thông tin đặc biệt, vLLM sử dụng Shared Memory để tránh sao chép dữ liệu qua lại giữa RAM tiến trình này sang RAM tiến trình kia.

---

## 2. Hoạt động của AsyncLLMEngine và RequestTracker

Tại tiến trình Frontend, khi người dùng gửi một request đến API Server, request này được đưa vào **`AsyncLLMEngine`** thông qua hàm `add_request()`.

```
[ HTTP Request ] 
       |
       v
[ FastAPI Endpoint ] 
       |
       v
[ AsyncLLMEngine.add_request() ]
       |
       +---> Khởi tạo AsyncStream (Trình sinh token bất đồng bộ)
       +---> Đưa vào RequestTracker.register_request()
                 |
                 v  (ZeroMQ Socket)
         [ EngineCoreServer ] (Nhận request và xếp vào Waiting Queue)
```

### RequestTracker:
* `RequestTracker` là nơi đăng ký và theo dõi vòng đời của tất cả các request đang hoạt động.
* Đối với mỗi request mới, `RequestTracker` tạo ra một đối tượng **`AsyncStream`**. Đây là một hàng đợi bất đồng bộ (`asyncio.Queue`) cục bộ.
* Khi tiến trình Backend sinh ra token mới cho request này, thông tin sẽ được đẩy qua ZeroMQ về tiến trình Frontend. Frontend nhận được sẽ detokenize (chuyển đổi ID token sang chữ) và đẩy trực tiếp vào `AsyncStream` tương ứng.
* API Endpoint chỉ việc thực hiện vòng lặp `async for` để đọc từ `AsyncStream` và trả dữ liệu về cho người dùng.

---

## 3. Quản lý Hàng đợi & Tránh nghẽn hàng đợi (Queue Starvation)

Tại tiến trình Backend, bộ lập lịch (`Scheduler`) quản lý các request thông qua 3 hàng đợi trạng thái:

1. **Waiting Queue**: Chứa các request mới gửi lên hoặc các request bị giải phóng bộ nhớ (Preempted bằng Recomputation) đang chờ được cấp phát khối bộ nhớ đầu tiên để chạy Prefill.
2. **Running Queue**: Chứa các request đang được cấp phát bộ nhớ GPU và đang chạy vòng lặp Decode sinh token.
3. **Swapped Queue**: Chứa các request bị tạm dừng tạm thời (Preempted bằng Swapping) và KV Cache của chúng đã bị đẩy sang RAM CPU.

### Thuật toán lập lịch xếp hàng (Scheduling Policy):
vLLM hỗ trợ nhiều chiến lược xếp hàng:
* **FCFS (First-Come-First-Served)**: Ưu tiên phục vụ request đến trước. Chiến lược này đơn giản và công bằng nhất.
* **Priority (Độ ưu tiên)**: Người dùng có thể gắn trọng số ưu tiên (`priority`) cho request. Bộ lập lịch sẽ sắp xếp hàng đợi theo mức ưu tiên này để phục vụ trước.

### Hiện tượng đói hàng đợi (Queue Starvation) và cách xử lý:
Nếu liên tục có request mới đến, hệ thống có thể bị cuốn vào việc chạy Prefill cho request mới mà "bỏ quên" các request đang Decode, hoặc ngược lại.
vLLM giải quyết việc này bằng cách:
* Luôn ưu tiên chạy các request trong **`Running Queue`** trước. Chỉ khi các request đang chạy được phân bổ đủ bộ nhớ, hệ thống mới xét đến việc đưa thêm request mới từ `Waiting Queue` vào.
* Giới hạn nghiêm ngặt số lượng sequence chạy song song (`max_num_seqs`) và giới hạn số token được tính toán trong mỗi step (`max_num_scheduled_tokens`) để đảm bảo không một request nào bị nghẽn quá lâu.

---

## 4. Kiến trúc Streaming SSE (Server-Sent Events)

Các ứng dụng Chat AI phổ biến (như ChatGPT, Claude) hiển thị chữ chạy ra từng từ một cách thời gian thực. Để làm được điều này, hệ thống Serving phải hỗ trợ giao thức **Streaming** thông qua **Server-Sent Events (SSE)**.

```
Client                      FastAPI Server (Frontend)             EngineCore (Backend)
  |                              |                                     |
  |--- POST /v1/chat/completions ->|                                     |
  |    (stream=True)             |                                     |
  |                              |--- Thêm Request qua ZMQ ------------>|
  |                              |                                     |
  |                              |                               [ Chạy Step 1 ]
  |<-- SSE (data: "Học") --------|--- Nhận token (ZMQ) ----------------|
  |                              |                                     |
  |                              |                               [ Chạy Step 2 ]
  |<-- SSE (data: " máy") -------|--- Nhận token (ZMQ) ----------------|
  |                              |                                     |
  |                              |                               [ Gặp token END ]
  |<-- SSE (data: [DONE]) -------|--- Báo hoàn thành (ZMQ) ------------|
```

Trong vLLM, khi cờ `stream=True` được kích hoạt:
1. HTTP Response được thiết lập Header: `Content-Type: text/event-stream`.
2. Ứng dụng API giữ kết nối HTTP luôn mở.
3. Ở mỗi bước lặp của GPU (`step()`), các token vừa sinh ra của tất cả các request được đóng gói và trả về Frontend.
4. FastAPI đọc từ `AsyncStream` của từng request và lập tức ghi (write) dữ liệu xuống cổng mạng HTTP dưới dạng format SSE: `data: {"choices": [{"delta": {"content": "..."}}]}`.

---

## 5. Xử lý Hủy kết nối đột ngột (Client Disconnection)

Một bài toán đau đầu trong thiết kế hệ thống Serving thực tế là **Client Disconnection** (Người dùng tắt tab trình duyệt hoặc hủy lệnh gọi API khi mô hình đang viết dở câu trả lời).
* **Vấn đề**: Nếu hệ thống không phát hiện ra sự kiện này, GPU vẫn sẽ tiếp tục chạy vô ích hàng trăm bước Decode để sinh token cho một request không còn ai nhận. Điều này làm lãng phí nghiêm trọng tài nguyên GPU VRAM và năng lực tính toán, trực tiếp làm chậm các request của người dùng khác.
* **Giải pháp của vLLM**:
  1. FastAPI Endpoint giám sát trạng thái kết nối mạng của HTTP Client.
  2. Nếu phát hiện client đóng kết nối (ví dụ: `await request.is_disconnected()` trả về `True`):
  3. `AsyncLLMEngine` lập tức kích hoạt lệnh **`abort_request(request_id)`**.
  4. Lệnh này được gửi qua ZeroMQ Socket đến Backend tiến trình `EngineCore`.
  5. Tại Backend, bộ lập lịch lập tức xóa request này khỏi `Running Queue` hoặc `Waiting Queue`.
  6. Bộ quản lý bộ nhớ (`KVCacheManager`) lập tức giải phóng tất cả các khối vật lý (Physical Blocks) đang chứa KV Cache của request này về lại Block Pool.

> [!IMPORTANT]
> Nhờ cơ chế hủy kết nối thời gian thực này, vLLM đảm bảo rằng 100% dung lượng GPU VRAM được sử dụng để phục vụ các yêu cầu thực tế đang hoạt động, loại bỏ hoàn toàn lãng phí tài nguyên do các kết nối bị bỏ rơi.

---

## 💡 Tổng kết bài học
* vLLM v1 sử dụng **kiến trúc tách biệt (Decoupled Engine)** chạy đa tiến trình giao tiếp qua ZeroMQ để tránh tắc nghẽn do GIL của Python.
* **RequestTracker** và **AsyncStream** là cầu nối bất đồng bộ nhận dữ liệu và cấp phát token trả về API Server cục bộ.
* Hệ thống quản lý hàng đợi chặt chẽ ngăn chặn hiện tượng **Queue Starvation** và tối ưu hóa thứ tự thực thi.
* **Streaming SSE** mang lại trải nghiệm người dùng mượt mà, kết hợp với cơ chế **Abort** thông minh tự động thu hồi tài nguyên VRAM khi client ngắt kết nối đột ngột.

Trong bài học tiếp theo, chúng ta sẽ bắt đầu đi sâu vào cấu trúc mã nguồn Python thực tế của vLLM để xem các lớp `Scheduler`, `KVCacheManager`, và `Request` được hiện thực như thế nào.
