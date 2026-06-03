# Bài 5: Deep Dive Codebase – Bộ lập lịch Scheduler, Request Queue & Block Manager

Trong các bài học trước, chúng ta đã nắm vững các khái niệm lý thuyết cốt lõi của vLLM. Từ bài học này, chúng ta sẽ trực tiếp đi sâu vào chi tiết hiện thực trong mã nguồn vLLM v1. Chúng ta sẽ cùng nhau bóc tách cách hoạt động của ba cấu phần cốt lõi: Bộ quản lý hàng đợi yêu cầu (`RequestQueue`), Bộ lập lịch (`Scheduler`), và Bộ quản lý KV Cache (`KVCacheManager`).

---

## 1. Quản lý Hàng đợi: `request_queue.py`

Hàng đợi chứa các request được lưu giữ trong file [request_queue.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/core/sched/request_queue.py). vLLM định nghĩa một Enum đại diện cho chiến lược lập lịch:

```python
class SchedulingPolicy(str, Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
```

Dựa trên cấu hình chính sách, vLLM sử dụng một factory `create_request_queue` để tạo ra đối tượng quản lý hàng đợi thích hợp:

```python
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    ...
```

* **`FCFSRequestQueue`**: Sử dụng cấu trúc dữ liệu `deque` (Double-ended Queue) để thực hiện chèn và xóa ở hai đầu với độ phức tạp $O(1)$. Thứ tự ưu tiên hoàn toàn dựa trên thời gian đến của request (`arrival_time`).
* **`PriorityRequestQueue`**: Lưu trữ các request sắp xếp theo thuộc tính `priority` của đối tượng `Request`. Khi có nhiều request có cùng mức ưu tiên, request nào đến trước (`arrival_time` nhỏ hơn) sẽ được xử lý trước.

---

## 2. Trái tim Lập lịch: `scheduler.py`

Bộ lập lịch `Scheduler` nằm tại [scheduler.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/core/sched/scheduler.py) là nơi điều phối chính của toàn bộ hệ thống. Nhiệm vụ chính của nó là hiện thực hàm `schedule()`, quyết định xem bước tiếp theo (iteration) GPU sẽ xử lý những token nào của những request nào.

### Các thuộc tính quan trọng của Lớp `Scheduler`:
* `self.waiting`: Hàng đợi chứa các request đang chờ xử lý prompt (kiểu `RequestQueue`).
* `self.running`: Danh sách chứa các request đang được suy luận trên GPU (kiểu `list[Request]`).
* `self.kv_cache_manager`: Đối tượng quản lý các khối KV Cache (kiểu `KVCacheManager`).

### Phân tích Luồng thực thi của hàm `schedule()`:

Hàm `schedule()` thực hiện việc gom lô động qua các bước tuần tự sau:

1. **Khởi tạo chu kỳ mới**:
   Gọi `self.kv_cache_manager.new_step_starts()` để báo cho trình quản lý bộ nhớ chuẩn bị cấp phát các khối cho bước suy luận tiếp theo.

2. **Lập lịch cho các Request đang chạy (`Running Queue`)**:
   Bộ lập lịch duyệt qua danh sách `self.running`. Với mỗi request đang chạy, nó tính toán số lượng token mới cần sinh (`num_new_tokens`).
   * Hệ thống tính toán xem có cần cấp phát thêm khối vật lý mới cho request này không bằng cách gọi `self.kv_cache_manager.allocate_slots(...)`.
   * **Xử lý cạn kiệt bộ nhớ (Preemption)**: Nếu hàm `allocate_slots` trả về `None` (tức cạn kiệt block trống trên GPU), bộ lập lịch sẽ thực hiện thu hồi tài nguyên bằng cách đẩy một hoặc nhiều request đang chạy có độ ưu tiên thấp nhất ra khỏi GPU:
     ```python
     # Mã nguồn đơn giản hóa của quá trình Preemption trong scheduler.py
     if new_blocks is None:
         # Chọn request để tạm dừng (preempt)
         preempted_req = self.running.pop() # Lấy request cuối (hoặc sắp xếp theo priority)
         self._preempt_request(preempted_req, scheduled_timestamp)
         preempted_reqs.append(preempted_req)
     ```
     Lệnh `_preempt_request` sẽ giải phóng các block tương ứng của request bị tạm dừng hoặc đẩy chúng sang RAM CPU qua cơ chế Swap.

3. **Lập lịch cho các Request đang chờ (`Waiting Queue`)**:
   Nếu sau khi lập lịch cho hàng đợi `running` mà hệ thống vẫn còn dư bộ nhớ GPU (khối vật lý khả dụng) và chưa vượt quá giới hạn token của batch (`token_budget`), bộ lập lịch sẽ duyệt tiếp đến hàng đợi `self.waiting`.
   * Lấy request ra từ hàng đợi chờ.
   * Tính toán xem request này có thể chia sẻ block cũ nào không (Prefix Caching).
   * Yêu cầu `KVCacheManager` cấp phát bộ nhớ KV Cache ban đầu.
   * Nếu thành công, đưa request này từ trạng thái `WAITING` sang `RUNNING` và thêm vào danh sách `self.running`.

4. **Đóng gói kết quả (`SchedulerOutput`)**:
   Trả về một đối tượng `SchedulerOutput` chứa thông tin chi tiết về các request được chọn chạy, số lượng token được lập lịch, bản đồ ánh xạ khối để gửi xuống cho Model Executor chạy trên GPU.

---

## 3. Bộ quản lý KV Cache: `kv_cache_manager.py`

Bộ quản lý bộ nhớ KV Cache thực tế nằm tại [kv_cache_manager.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/core/kv_cache_manager.py). Đây là cấu phần tương tự như quản lý bộ nhớ của Hệ điều hành.

```
+-----------------------------------------------------------+
|                      KVCacheManager                       |
|  - BlockPool (Danh sách tất cả các khối vật lý trống)      |
|  - BlockAllocator (Cấp phát khối mới)                      |
|  - PageTable (Bản đồ ánh xạ request_id -> list[PhysBlock]) |
+-----------------------------------------------------------+
```

### Các nhiệm vụ chính của `KVCacheManager`:
1. **Quản lý Vòng đời Block (Block Lifecycle)**:
   * Giữ một danh sách các khối vật lý trống (`free_list`).
   * Khi một request yêu cầu cấp phát bộ nhớ cho token mới, `KVCacheManager` sẽ rút một khối vật lý từ `free_list` ra và ánh xạ nó vào danh sách khối logic của request trong bảng `PageTable`.
   * Khi một request hoàn thành hoặc bị Reempt, tất cả các khối vật lý của nó được trả lại về `free_list`.

2. **Cơ chế Copy-on-Write (CoW)**:
   * Khi thực hiện chia sẻ block (ví dụ trong Parallel Sampling), `KVCacheManager` tăng bộ đếm tham chiếu (`ref_count`) của khối vật lý chung đó lên.
   * Khi một request ghi đè hoặc sinh token mới trên khối dùng chung đó, `KVCacheManager` nhận diện `ref_count > 1`, tự động sao chép khối đó ra một địa chỉ vật lý mới, cập nhật lại `PageTable` cho request ghi, và giảm `ref_count` khối cũ đi 1.

---

## 4. Vòng lặp thực thi của Engine: `core.py`

Tất cả các thành phần trên được kết nối và vận hành liên tục bởi lớp `EngineCore` trong file [core.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/engine/core.py).

Hàm `step()` trong `EngineCore` chạy một vòng lặp liên tục ở tiến trình Backend:

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    # 1. Kiểm tra xem scheduler có request nào không
    if not self.scheduler.has_requests():
        return {}, False
    
    # 2. Gọi Scheduler để gom lô và lập lịch
    scheduler_output = self.scheduler.schedule()
    
    # 3. Đẩy thông tin lập lịch xuống Model Executor để chạy forward trên GPU (non-blocking)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    
    # 4. Đọc kết quả tính toán (logits) và chạy sampler để lấy token ID mới sinh ra
    model_output = future.result()
    
    # 5. Cập nhật kết quả sinh token ngược lại cho Scheduler để cập nhật trạng thái các request
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

> [!NOTE]
> **Nhận xét**: Sự kết hợp chặt chẽ giữa `scheduler.py` (Lập lịch thông minh), `kv_cache_manager.py` (Quản lý bộ nhớ hiệu quả), và `core.py` (Động cơ thực thi đồng bộ/bất đồng bộ) tạo nên sức mạnh tối ưu hóa vượt trội của vLLM. Mỗi thành phần đều tập trung giải quyết bài toán hiệu năng ở cấp độ phần cứng cao nhất.

---

## 💡 Tổng kết bài học
* `RequestQueue` quản lý thứ tự ưu tiên của các request đến hệ thống (FCFS hoặc Priority).
* `Scheduler` thực hiện thuật toán lập lịch mức Iteration chính, chịu trách nhiệm gom batch động, xử lý preemption để bảo vệ bộ nhớ GPU khỏi bị sập do OOM.
* `KVCacheManager` quản lý cấp phát động bộ nhớ GPU dưới dạng các khối kích thước cố định, quản lý bảng ánh xạ trang và thực hiện Copy-on-Write khi chia sẻ dữ liệu.
* `EngineCore` vận hành vòng lặp `step()` để điều phối toàn bộ các cấu phần trên hoạt động ăn khớp với GPU Workers.

Trong bài tiếp theo, chúng ta sẽ chuyển sang khảo sát phân hệ thực thi mô hình: **Distributed Executor, GPU Worker và cơ chế CUDA Graphs**.
