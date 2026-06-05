---
sidebar_position: 8.92
sidebar_label: "Bài 6.5: Phục vụ MoE và Cân bằng tải EPLB"
---

# Bài 6.5: Phục vụ MoE trên Multi-GPU - Expert Parallelism và Cân bằng tải EPLB

Sự bùng nổ của các mô hình sử dụng cấu trúc hỗn hợp chuyên gia (Mixture of Experts - MoE) như DeepSeek V3/V4 hay Mixtral 8x7B đặt ra những thách thức chưa từng có cho hệ thống phục vụ đa GPU. MoE sở hữu lượng tham số khổng lồ nhưng chỉ kích hoạt một phần nhỏ cho mỗi token, dẫn đến việc phân bổ bộ nhớ weights và tính toán trên đa GPU trở nên vô cùng phức tạp.

Bài học này sẽ giúp bạn hiểu sâu về cơ chế song song hóa chuyên gia (Expert Parallelism), nguồn gốc của vấn đề lệch tải chuyên gia gây nghẽn GPU, và giải thuật **EPLB (Expert Parallel Load Balancer)** được tích hợp trong vLLM để giải quyết triệt để bài toán này.

---

## 1. Cơ chế song song hóa MoE: Tensor Parallelism vs Expert Parallelism

Khi chạy phục vụ một mô hình MoE trên đa GPU (ví dụ 4x GPU), chúng ta phải kết hợp hai phương pháp song song hóa khác nhau:

### A. Tensor Parallelism (TP) cho tầng Attention
Tầng Self-Attention được chia sẻ chung bởi tất cả các tokens. Tầng này được song song hóa bằng **Tensor Parallelism (TP)**, nơi các ma trận trọng số Projection (Q, K, V, O) được cắt lát và phân bổ đều trên cả 4 GPU. Ở cuối tầng Attention, hệ thống bắt buộc phải chạy một phép giao tiếp **All-Reduce** để đồng bộ hóa các tensor.

### B. Expert Parallelism (EP) cho tầng MoE
Tầng MLP/MoE chứa nhiều chuyên gia độc lập (ví dụ 8 hoặc 256 experts). Thay vì cắt ma trận của từng expert (điều này gây ra overhead giao tiếp All-Reduce quá lớn cho các mô hình nhỏ), chúng ta phân chia các expert nguyên vẹn cho các GPU. Ví dụ với mô hình có 8 experts chạy trên 4 GPU:
*   GPU 0: Giữ Expert 1, 2
*   GPU 1: Giữ Expert 3, 4
*   GPU 2: Giữ Expert 5, 6
*   GPU 3: Giữ Expert 7, 8

```
[ Input Tokens ] ──> [ Router ]
                        │
      ┌─────────────────┼─────────────────┐
      ▼ (All-to-All)    ▼ (All-to-All)    ▼ (All-to-All)
   [ GPU 0 ]         [ GPU 1 ]         [ GPU 2 ]
  Expert 1, 2       Expert 3, 4       Expert 5, 6
```

### Overhead giao tiếp All-to-All
Khi các token đi qua tầng Router, Router sẽ quyết định gửi từng token đến đúng GPU chứa expert mà token đó lựa chọn. Quá trình gửi và nhận chéo các token giữa các GPU này được thực hiện thông qua phép giao tiếp mạng **All-to-All** của NCCL. Đây là phép giao tiếp phức tạp, yêu cầu băng thông liên kết GPU cực cao.

---

## 2. Vấn đề lệch tải chuyên gia (Expert Imbalance)

Trong môi trường phục vụ thực tế (Production serving), sự phân bổ các token cho các expert thường **không bao giờ đồng đều**. 

### Hotspot Experts (Điểm nóng chuyên gia)
Tùy thuộc vào nội dung câu hỏi của người dùng, một số chuyên gia nhất định sẽ bị kích hoạt liên tục với tần suất rất cao (ví dụ: chuyên gia chuyên xử lý cú pháp lập trình, chuyên gia chuyên về logic toán học hoặc các từ khóa ngữ pháp tiếng Anh phổ biến). Ngược lại, các chuyên gia chuyên biệt khác gần như không được gọi đến.

### Hậu quả của lệch tải trên GPU:
Giả sử GPU 0 đang nắm giữ hai chuyên gia "hot" nhất là Expert 1 và Expert 2. Khi đó:
1.  **GPU 0 quá tải**: Phải thực hiện tính toán forward MLP liên tục cho một lượng lớn token được định tuyến đến.
2.  **GPU 1, 2, 3 rảnh rỗi**: Phải đứng chờ GPU 0 tính toán xong để đồng bộ hóa bước tiếp theo.
3.  **Nghẽn cổ chai**: Tốc độ sinh token của toàn bộ batch bị kéo lùi về tốc độ của GPU chạy chậm nhất (GPU 0). Hiệu năng hệ thống phục vụ lúc này bị suy giảm nghiêm trọng, triệt tiêu lợi ích song song hóa.

---

## 3. Giải thuật EPLB (Expert Parallel Load Balancer) của vLLM

Để giải quyết triệt để vấn đề lệch tải chuyên gia, vLLM v1 tích hợp lớp điều phối `EPLBController` thông qua tệp [eplb_utils.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/worker/gpu/eplb_utils.py), phối hợp chặt chẽ với [model_runner.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/worker/gpu/model_runner.py) và [gpu_worker.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/worker/gpu_worker.py).

EPLB thay đổi tư duy từ phân bổ expert tĩnh sang **phân bổ expert động** dựa trên việc đo đạc tải thực tế của hệ thống.

```
       [ EPLB Controller ]
               │ (Giám sát tần suất kích hoạt Experts)
               ▼
┌─────────────────────────────────────────┐
│ Giải pháp 1: Replicate (Nhân bản)       │
│ - Copy Expert "hot" sang nhiều GPU      │
│   (ví dụ: Expert 1 nằm trên cả GPU 0&1)  │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ Giải pháp 2: Bin-Packing (Đóng gói)     │
│ - Gộp các Expert ít dùng vào chung GPU  │
│   để tiết kiệm VRAM cho nhân bản        │
└─────────────────────────────────────────┘
```

### Các cơ chế tối ưu của giải thuật EPLB:

#### A. Giám sát động (Dynamic Profiling)
Ở mỗi bước chạy thử hoặc trong quá trình phục vụ thực tế, `EPLBController` thu thập thống kê về tần suất định tuyến của Router đến các expert. Từ đó xác định được "bản đồ phân bổ tải" (workload distribution map) của các chuyên gia.

#### B. Nhân bản chuyên gia nóng (Expert Replication)
Đối với các expert được kích hoạt quá nhiều vượt quá một ngưỡng tải quy định, EPLB sẽ quyết định **nhân bản (replicate)** expert đó sang nhiều GPU khác nhau.
*   *Ví dụ*: Nếu Expert 1 cực kỳ hot, EPLB sẽ copy trọng số của Expert 1 lên cả GPU 0 và GPU 1.
*   Khi chạy forward, Router có thể gửi một nửa số token cần Expert 1 đến GPU 0 và nửa còn lại đến GPU 1, giúp chia đôi tải tính toán và loại bỏ hoàn toàn hiện tượng hotspot.

#### C. Đóng gói chuyên gia lạnh (Bin-Packing Expert Placement)
Việc nhân bản expert sẽ tiêu tốn thêm bộ nhớ VRAM của GPU. Để cân đối tài nguyên, EPLB áp dụng thuật toán đóng gói thùng (Bin-packing) cho các expert ít sử dụng:
*   Gộp nhiều expert rảnh rỗi vào chung một GPU để nhường không gian VRAM trên các GPU khác cho việc nhân bản các expert nóng.

---

## 4. Tích hợp EPLB trong ModelRunner và GPU Worker

Hãy cùng xem luồng code tích hợp EPLB chạy trong vòng lặp thực thi của vLLM v1.

### Khởi tạo EPLB
Khi [ModelRunner](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/worker/gpu/model_runner.py) khởi chạy, nó sẽ kiểm tra cấu hình song song và khởi tạo bộ điều phối EPLB:

```python
# Trích từ model_runner.py: __init__()
from vllm.v1.worker.gpu.eplb_utils import EPLBController, step_eplb_after

self.eplb = EPLBController(self.parallel_config, self.device)
```

### Cập nhật tải sau mỗi bước chạy (Step Callback)
vLLM sử dụng decorator Python `@step_eplb_after()` để tự động cập nhật trạng thái cân bằng tải ngay sau khi một lượt forward hoặc verify hoàn tất:

```python
# Trích từ eplb_utils.py:
def step_eplb_after(*, is_dummy: bool = False) -> Callable:
    """Tự động gọi EPLB step sau khi model runner thực thi xong"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            # Kích hoạt tính toán lại phân bổ expert nếu cần thiết
            self.eplb.step(is_dummy=is_dummy, is_profile=is_profile)
            return res
        return wrapper
    return decorator
```

Sự kết hợp chặt chẽ này đảm bảo rằng sơ đồ phân bổ expert luôn được tối ưu hóa liên tục theo thời gian thực dựa trên hành vi prompt thực tế của người dùng, giúp hệ thống MoE đa GPU luôn duy trì được thông lượng (throughput) tối đa.

---

## 5. Liên hệ với Toy Engine: Mô hình đồng nhất vs Hỗn hợp chuyên gia (MoE)

Hãy cùng đối chiếu cách xử lý tính toán giữa mô hình mô phỏng của chúng ta và thực tế chạy MoE:

*   **Toy Engine (Mô hình đồng nhất - Homogeneous)**: Trong [model.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/toy_engine/model.py), chúng ta hiện thực hóa `MockModel` với một giả định đơn giản: mọi token sinh ra ở pha Decode đều tốn một lượng thời gian tính toán hoàn toàn bằng nhau (được giả lập bằng hàm `asyncio.sleep` với độ trễ cố định). Điều này đại diện cho các mô hình Dense truyền thống (như Llama-3-8B).
*   **Production vLLM (MoE và Lệch tải)**: Đối với mô hình MoE thực tế như DeepSeek V3, mỗi token đi qua Router sẽ kích hoạt các chuyên gia (experts) khác nhau. Nếu chúng ta đưa MoE vào `toy_engine`, thời gian xử lý của từng request sẽ không còn đồng đều mà biến thiên phụ thuộc vào việc token đó kích hoạt expert nào. Trên môi trường đa GPU, điều này tạo ra hiện tượng lệch tải (GPU này tính toán MLP nặng nề trong khi GPU kia rảnh rỗi chờ đợi). `EPLBController` sinh ra để điều hòa sự bất đối xứng này bằng cách di chuyển và nhân bản các expert nóng, đảm bảo hiệu năng GPU đồng đều.

---

## 💡 Tổng kết bài học

*   **Expert Parallelism (EP)** phân chia các expert của mô hình MoE lên các GPU khác nhau, giao tiếp chéo giữa các GPU qua NCCL **All-to-All**.
*   **Expert Imbalance** xảy ra khi các token định tuyến tập trung vào một số expert nóng (hotspots), khiến GPU chứa expert đó bị quá tải và bắt các GPU khác phải chờ rảnh rỗi.
*   **vLLM EPLB** giải quyết triệt để lệch tải bằng cách giám sát động tần suất dùng expert, tự động **nhân bản (replicate)** các expert nóng sang nhiều GPU để chia sẻ tải và **đóng gói (bin-packing)** các expert ít dùng để tối ưu hóa bộ nhớ VRAM.
