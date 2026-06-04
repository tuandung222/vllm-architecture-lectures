---
layout: default
title: "Bài 0: Kiến thức Hệ điều hành bổ trợ (OS Fundamentals for AI Serving)"
---

# Bài 0: Kiến thức Hệ điều hành bổ trợ (OS Fundamentals for AI Serving)

Để hiểu sâu sắc các quyết định thiết kế kiến trúc của vLLM (như PagedAttention, Decoupled Engine, Multi-processing Workers), chúng ta cần trang bị nền tảng kiến thức vững chắc về **Hệ điều hành (Operating System - OS)**. 

Bài viết này cung cấp background chi tiết về hai chủ đề cốt lõi: **Cơ chế Phân trang Bộ nhớ (Paging)** và **Giao tiếp giữa các Tiến trình (IPC, Shared Memory, ZeroMQ)** dưới góc nhìn của một Kỹ sư AI Serving.

---

## 1. Phân trang Bộ nhớ (Memory Paging & Page Tables)

Trong các máy tính hiện đại, các chương trình không truy cập trực tiếp vào RAM vật lý. Thay vào đó, chúng tương tác với một lớp trừu tượng gọi là **Bộ nhớ ảo (Virtual Memory)**.

### 1.1. Bộ nhớ ảo (Virtual Memory) vs Bộ nhớ vật lý (Physical Memory)
* **Bộ nhớ vật lý**: Là các chip RAM thực tế cắm trên bo mạch chủ, có địa chỉ vật lý cố định từ `0x00000000` đến hết dung lượng RAM.
* **Bộ nhớ ảo**: Mỗi tiến trình (Process) khi khởi chạy được hệ điều hành cấp cho một không gian địa chỉ ảo liên tục, độc lập (ví dụ: từ `0` đến $2^{64}-1$ trên hệ điều hành 64-bit).
* **Mục đích**: 
  * **Bảo vệ (Protection)**: Ngăn không cho tiến trình A đọc/ghi đè lên bộ nhớ của tiến trình B.
  * **Tiện lợi**: Lập trình viên có thể viết code giả định rằng chương trình có một vùng nhớ liên tục khổng lồ mà không cần quan tâm RAM thực tế còn trống ở đâu.

### 1.2. Phân trang (Paging) là gì?
Hệ điều hành chia cả bộ nhớ ảo và bộ nhớ vật lý thành các khối nhỏ có kích thước cố định bằng nhau (thường là 4KB trên các hệ thống x86):
* **Trang logic (Pages)**: Các khối thuộc không gian bộ nhớ ảo của tiến trình.
* **Khung trang vật lý (Page Frames)**: Các khối thuộc RAM vật lý.

```
Không gian Địa chỉ ảo (Virtual Address Space):
[ Page 0 (4KB) ] -> [ Page 1 (4KB) ] -> [ Page 2 (4KB) ] -> [ Page 3 (4KB) ]
       |                  |                  |                  |
       v (Ánh xạ)         v (Ánh xạ)         v (Ánh xạ)         v (Ánh xạ)
[ Frame 42 (RAM) ] [ Frame 12 (RAM) ] [ Frame 99 (RAM) ] [ Frame 03 (RAM) ]
```

### 1.3. Bảng trang (Page Table) & MMU (Memory Management Unit)
* **Page Table**: Là một cấu trúc dữ liệu lưu trong RAM, do Hệ điều hành quản lý, ghi nhận ánh xạ: *Page $X$ của tiến trình này đang nằm ở Frame vật lý $Y$ nào trong RAM*.
* **MMU**: Là một bộ phận phần cứng nằm trong CPU. Mỗi khi CPU thực hiện lệnh đọc/ghi bộ nhớ ở địa chỉ ảo, MMU sẽ tự động tra cứu Page Table để dịch địa chỉ ảo đó thành địa chỉ vật lý trong RAM thời gian thực.
* **Phân mảnh ngoại bộ (External Fragmentation)**: Nhờ có phân trang, hệ điều hành có thể tận dụng các vùng RAM vật lý trống nằm rải rác ở khắp nơi để cấp phát cho một tiến trình cần vùng nhớ liên tục. Điều này loại bỏ hoàn toàn hiện tượng phân mảnh ngoại bộ.

### 1.4. Mối tương quan trực tiếp với vLLM:
vLLM đã bê nguyên xi thiết kế này từ OS vào quản lý VRAM của GPU:

| Khái niệm trong Hệ điều hành (OS) | Tương đương trong vLLM |
| :--- | :--- |
| **Virtual Memory (Bộ nhớ ảo)** | **Logical KV Cache** (Chuỗi token liên tục của request) |
| **Physical Memory (RAM vật lý)** | **GPU VRAM Block Pool** (Các vùng nhớ thực tế trên GPU) |
| **Pages (Trang)** | **Logical Blocks** (Khối logic chứa 16/32 tokens) |
| **Page Frames (Khung trang)** | **Physical Blocks** (Khối vật lý thực tế trên VRAM) |
| **Page Table (Bảng trang)** | **Block Mapping Table** (Bảng ánh xạ khối của request) |
| **OS Memory Manager** | **KVCacheManager** (Quản lý cấp phát, thu hồi và swap) |

---

## 2. Giao tiếp giữa các Tiến trình (Inter-Process Communication - IPC)

Mỗi tiến trình Python chạy trên OS đều có một không gian bộ nhớ ảo hoàn toàn tách biệt. Tiến trình API Server không thể đọc trực tiếp các biến hay tensor nằm trong tiến trình GPU Worker. Để trao đổi dữ liệu, chúng bắt buộc phải sử dụng các cơ chế **IPC**.

```
 Tiến trình A (Frontend)                    Tiến trình B (Backend GPU)
 ┌──────────────────────┐                  ┌──────────────────────┐
 │ Không gian nhớ ảo A  │                  │ Không gian nhớ ảo B  │
 └──────────┬───────────┘                  └───────────┬──────────┘
            │                                          │
            └───────────────> [ Cầu IPC ] <────────────┘
                        (Shared Memory / Sockets)
```

Có nhiều giải pháp IPC, nhưng trong AI Serving hiệu năng cao, chúng ta chỉ tập trung vào hai cơ chế tối ưu nhất: **Shared Memory** và **ZeroMQ**.

### 2.1. Shared Memory (Bộ nhớ chia sẻ) - IPC nhanh nhất
* **Nguyên lý**: Hệ điều hành cho phép hai hoặc nhiều tiến trình cùng ánh xạ một vùng RAM vật lý (hoặc VRAM) chung vào không gian địa chỉ ảo của riêng mình.
* **Đặc điểm**:
  * Khi tiến trình A ghi dữ liệu vào vùng nhớ này, tiến trình B lập tức nhìn thấy thay đổi ngay lập tức mà không cần bất kỳ lệnh gọi hệ thống (system call) hay sao chép dữ liệu (zero-copy) nào qua mạng.
  * Tốc độ cực nhanh, bị giới hạn duy nhất bởi băng thông của thanh RAM.
* **Ứng dụng trong AI Serving**: Khi chạy song song mô hình trên nhiều GPU (Tensor Parallelism), các tiến trình Worker cần truyền các tensor dữ liệu kích thước lớn cho nhau. vLLM sử dụng **Shared Memory IPC** để chia sẻ các tensor này cực kỳ nhanh mà không bị nghẽn cổ chai I/O.

### 2.2. ZeroMQ (ZMQ) & IPC Sockets - Truyền nhận thông điệp hiệu năng cao
Khi cần truyền thông điệp điều khiển (như Metadata của request, trạng thái lập lịch, token ID vừa sinh), Shared Memory quá phức tạp để đồng bộ hóa (cần dùng mutex, lock). Thay vào đó, hệ thống sử dụng cơ chế truyền tin (Message Passing) qua **ZeroMQ**.

* **ZeroMQ (ZMQ) là gì?**: ZMQ không phải là một Message Broker (như RabbitMQ hay Kafka), mà là một thư viện Socket bất đồng bộ siêu nhanh, cung cấp các cổng giao tiếp trực tiếp giữa các tiến trình mà không cần server trung gian.
* **Tại sao vLLM dùng ZMQ thay vì TCP Socket truyền thống?**:
  1. **Tự động hàng đợi (Built-in Queue)**: ZMQ tự động quản lý hàng đợi gửi/nhận tin nhắn ở tầng dưới bằng luồng chạy ngầm. Tiến trình gửi không bị chặn (non-blocking) khi tiến trình nhận đang bận xử lý.
  2. **Tự động kết nối lại (Auto-reconnection)**: Nếu tiến trình GPU Worker bị khởi động lại, ZMQ tự động thiết lập lại kết nối mà không làm crash hệ thống API.
  3. **Hỗ trợ IPC Socket**: ZMQ hỗ trợ giao thức `ipc://` truyền dữ liệu trực tiếp qua kernel của hệ điều hành mà không cần đi qua giao thức mạng TCP/IP loopback (`127.0.0.1`), giúp giảm độ trễ tối đa.

---

## 3. Các mô hình giao tiếp ZMQ phổ biến trong vLLM

vLLM sử dụng các Socket Pattern (Mô hình kết nối) của ZMQ để phân phối tải và đồng bộ hóa:

### 3.1. Mô hình Router - Dealer (Đồng thời & Bất đồng bộ)
Được sử dụng giữa **Frontend Process** (FastAPI/AsyncLLMEngine) và **Backend Process** (EngineCore).

```
   [ Client 1 ] ---\             ┌──────────────┐             /---> [ GPU Worker 0 ]
   [ Client 2 ] ----> ZMQ ROUTER ┤  ZMQ DEALER  ├ ZMQ DEALER ----> [ GPU Worker 1 ]
   [ Client 3 ] ---/             └──────────────┘             \---> [ GPU Worker 2 ]
```

* **ROUTER (Frontend)**: Nhận nhiều request từ các client khác nhau đồng thời. Nó tự động gán một nhãn định danh (Identity Envelope) cho mỗi thông điệp để biết tin nhắn đó đến từ kết nối nào.
* **DEALER (Backend)**: Nhận các thông điệp từ Router và phân phối bất đồng bộ cho vòng lặp Engine xử lý. Khi Engine sinh xong token, Dealer gửi ngược lại cho Router kèm theo nhãn định danh để Router chuyển đúng về cho API stream của client đó.
* **Lợi ích**: Cho phép truyền nhận hai chiều song song, không chặn luồng (fully asynchronous), giúp hệ thống phục vụ hàng ngàn request đồng thời.

### 3.2. Mô hình Push - Pull (Đường ống xử lý tuần tự)
Được sử dụng để đẩy các chỉ thị hoặc logs/metrics từ GPU Workers về cho bộ phận ghi log trung tâm.
* **PUSH**: Tiến trình gửi đẩy thông điệp vào đường ống. Nếu hàng đợi đầy, nó sẽ xếp hàng chờ.
* **PULL**: Tiến trình nhận kéo thông điệp ra để xử lý theo thứ tự đến trước xử lý trước.

---

## 💡 Tóm tắt kiến thức cốt lõi cho Kỹ sư AI Serving
1. **Phân trang bộ nhớ (Paging)** giải quyết bài toán **phân mảnh** bằng cách chia nhỏ không gian nhớ thành các khối bằng nhau và ánh xạ linh hoạt qua bảng trang. vLLM dùng nguyên lý này để tạo ra PagedAttention.
2. **Tiến trình độc lập** có vùng nhớ ảo cách biệt để bảo vệ lẫn nhau. Do đó, để làm serving đa tiến trình, ta bắt buộc phải dùng **IPC**.
3. **Shared Memory** là con đường nhanh nhất để chia sẻ Tensor dung lượng lớn giữa các GPU GPU Workers mà không tốn chi phí copy.
4. **ZeroMQ** là khung xương giao tiếp điều khiển siêu nhanh của vLLM v1, giúp tách biệt hoàn toàn API Server bất đồng bộ (FastAPI) khỏi nhân tính toán đồng bộ của GPU (EngineCore) qua mô hình **Router-Dealer**.
