---
sidebar_position: 8.5
sidebar_label: "Bài 6.2: Song song Tensor & Giao tiếp NCCL"
---

# Bài 6.2: Phân tích Kỹ thuật Song song Tensor (Tensor Parallelism) & Giao tiếp NCCL

Khi phục vụ các mô hình ngôn ngữ lớn (LLM), kích thước bộ nhớ và năng lực tính toán của một GPU đơn lẻ thường không đủ đáp ứng. Trong [Bài 6](./lesson_6_distributed_worker.md), chúng ta đã tìm hiểu sơ bộ về các cơ chế song song hóa.

Bài học này sẽ đi sâu vào toán học phân rã ma trận của **Megatron-LM Tensor Parallelism (TP)**, cơ chế truyền thông mạng tốc độ cao **NCCL All-Reduce**, và cách lập lịch **Pipeline Parallelism (PP)** tránh thời gian chết (bubbles).

---

## 1. Megatron-LM Tensor Parallelism: Column vs Row Parallel

Cơ chế song song hóa Tensor (TP) chia nhỏ các lớp tuyến tính (Linear Layers) của Transformer và phân bổ cho nhiều GPU vật lý trong cùng một nhóm truyền thông (TP Group). vLLM kế thừa thiết kế kinh điển của **Megatron-LM** với hai thành phần cốt lõi: `ColumnParallelLinear` và `RowParallelLinear`.

```
              ┌───────────────────────────┐
              │    Input Tensor X (Replicated)
              └─────────────┬─────────────┘
             ┌──────────────┴──────────────┐
             ▼                             ▼
   ┌──────────────────┐          ┌──────────────────┐
   │ GPU 0: Col-Split │          │ GPU 1: Col-Split │  (ColumnParallelLinear)
   │    Y0 = X * W1   │          │    Y1 = X * W2   │
   └─────────┬────────┘          └─────────┬────────┘
             │                             │
             ▼                             ▼
   ┌──────────────────┐          ┌──────────────────┐
   │ GPU 0: Row-Split │          │ GPU 1: Row-Split │  (RowParallelLinear)
   │   Z0 = Y0 * V1   │          │   Z1 = Y1 * V2   │
   └─────────┬────────┘          └─────────┬────────┘
             └──────────────┬──────────────┘
                            │  NCCL All-Reduce
                            ▼
              ┌───────────────────────────┐
              │   Final Output Z = Z0 + Z1
              └───────────────────────────┘
```

### 1.1. Lớp song song theo cột: `ColumnParallelLinear`
Được áp dụng cho các lớp chiếu đầu vào: QKV Projection trong khối Attention, hoặc Gate & Up Projections trong khối MLP (SwiGLU).

*   **Phân rã ma trận**: Ma trận trọng số $W$ được cắt theo các cột thành $N$ phần tương ứng với $N$ GPU:
    $$W = [W_1 \mid W_2 \mid \dots \mid W_N]$$
*   **Thực thi**:
    *   Tensor đầu vào $X$ được nhân bản (replicate) trên toàn bộ $N$ GPU.
    *   Mỗi GPU $i$ thực hiện phép nhân ma trận độc lập:
        $$Y_i = X \cdot W_i$$
    *   **Không có giao tiếp mạng nào** cần thiết trong quá trình tính toán này. Dữ liệu đầu ra $Y_i$ tạm thời được giữ ở trạng thái phân chia trên mỗi GPU.

### 1.2. Lớp song song theo dòng: `RowParallelLinear`
Được áp dụng cho các lớp chiếu đầu ra: Out Projection trong khối Attention, hoặc Down Projection trong khối MLP.

*   **Phân rã ma trận**: Ma trận trọng số $V$ được cắt theo các dòng thành $N$ phần tương ứng với $N$ GPU:
    $$V = \begin{bmatrix} V_1 \\ V_2 \\ \dots \\ V_N \end{bmatrix}$$
*   **Thực thi**:
    *   Tensor đầu vào của lớp này chính là đầu ra phân chia từ bước trước: $Y = [Y_1 \mid Y_2 \mid \dots \mid Y_N]$. Mỗi GPU $i$ chỉ giữ một phần dữ liệu $Y_i$.
    *   Mỗi GPU $i$ thực hiện nhân ma trận dòng độc lập:
        $$Z_i = Y_i \cdot V_i$$
    *   Để thu được kết quả chính xác cuối cùng của lớp:
        $$Z = X \cdot W \cdot V = \sum_{i=1}^{N} Y_i \cdot V_i = \sum_{i=1}^{N} Z_i$$
    *   Để thực hiện phép cộng tổng ($\sum$) các tensor cục bộ từ tất cả các GPU, hệ thống kích hoạt một phép toán truyền thông đồng bộ **All-Reduce (Sum)** trên nhóm GPU.

> [!TIP]
> **Tính tối ưu của Thiết kế**: Bằng cách ghép cặp liên tiếp `ColumnParallelLinear` ➔ `RowParallelLinear` (ví dụ Gate/Up ➔ Down trong MLP), hệ thống chỉ cần kích hoạt **duy nhất 1 phép toán All-Reduce** ở cuối khối MLP, loại bỏ hoàn toàn giao tiếp mạng trung gian.

---

## 2. Bản chất Giao tiếp NCCL: Thuật toán Ring-based All-Reduce

Phép toán **All-Reduce** là nút thắt cổ chai truyền thông chính của Tensor Parallelism. vLLM sử dụng thư viện **NCCL (NVIDIA Collective Communications Library)** để thực thi phép toán này một cách tối ưu ở tầng phần cứng qua cầu nối NVLink.

Đối với các Tensor dữ liệu có kích thước lớn, NCCL sử dụng giải thuật **Ring-based All-Reduce** để tối đa hóa băng thông đường truyền.

### 2.1. Giải thuật Ring-based All-Reduce
Giả sử chúng ta có $N$ GPU xếp thành một vòng tròn khép kín (Ring). Mỗi GPU chỉ giao tiếp trực tiếp với GPU liền kề (GPU $i$ gửi cho GPU $i+1$).

```
       ┌───────┐      ┌───────┐
  ───> │ GPU 0 │ ───> │ GPU 1 │ ───>
       └───────┘      └───────┘
           ▲              │
           │              ▼
       ┌───────┐      ┌───────┐
  <─── │ GPU 3 │ <─── │ GPU 2 │ <───
       └───────┘      └───────┘
```

1.  **Chia nhỏ dữ liệu**: Tensor cần All-Reduce (kích thước $S$ Bytes) được chia thành $N$ phần (chunks) bằng nhau.
2.  **Pha 1: Reduce-Scatter ($N-1$ bước)**:
    *   Ở mỗi bước, mỗi GPU gửi một chunk dữ liệu của mình cho GPU tiếp theo, đồng thời nhận một chunk từ GPU phía trước và cộng dồn (Sum) vào chunk tương ứng của mình.
    *   Sau $N-1$ bước, mỗi GPU sẽ giữ một chunk chứa kết quả cộng dồn cuối cùng của toàn bộ hệ thống.
3.  **Pha 2: All-Gather ($N-1$ bước)**:
    *   Mỗi GPU bắt đầu gửi chunk đã cộng dồn hoàn chỉnh của mình đi quanh vòng tròn để đồng bộ cho các GPU khác.
    *   Sau $N-1$ bước tiếp theo, tất cả các GPU đều nhận đủ $N$ chunks đã cộng dồn, hoàn thành All-Reduce.

### 2.2. Phân tích Băng thông (Bandwidth Analysis)
Tổng dung lượng dữ liệu truyền qua mạng của mỗi GPU trong suốt quá trình Ring All-Reduce là:

$$\text{Data Sent} = 2 \times \frac{N-1}{N} \times S \text{ Bytes}$$

*   Khi số lượng GPU $N$ lớn, hệ số $2 \frac{N-1}{N}$ tiệm cận về **$2$**.
*   **Nhận xét**: Tổng dung lượng truyền tải trên mỗi GPU gần như **độc lập** với số lượng GPU $N$. Điều này giúp Ring All-Reduce có khả năng mở rộng (scalability) cực kỳ tốt khi tăng số lượng GPU trong hệ thống.

---

## 3. Pipeline Parallelism (PP) & Quản lý KV Cache liên node

Trong Pipeline Parallelism (PP), mô hình được chia theo chiều dọc (các layer). Ví dụ: Mô hình 32 layers chạy trên 2 GPU: GPU 0 xử lý layer 1-16, GPU 1 xử lý layer 17-32.

### 3.1. Luồng Lập lịch 1F1B (One Forward, One Backward) trong Training vs Serving
*   Trong quá trình huấn luyện (Training), PP gặp vấn đề lớn về bộ nhớ do phải lưu giữ các kích hoạt (activations) để tính toán gradient ở pha Backward.
*   Trong quá trình suy luận (Serving/Inference), **không có pha Backward**. Tuy nhiên, PP vẫn gặp vấn đề **Pipeline Bubble (Thời gian bong bóng/Thời gian rảnh)**: GPU 1 phải ngồi đợi GPU 0 xử lý xong layer 16 mới có dữ liệu để tính toán.

```
Không tối ưu (Sequential PP):
GPU 0: [ Fwd 1 ] --------------------> [ Fwd 2 ] -------------------->
GPU 1:           [ Fwd 1 ] (Đợi GPU 0)           [ Fwd 2 ] (Đợi GPU 0)
Bubble:          ^^^^^^^^^                       ^^^^^^^^^  (Lãng phí GPU)
```

vLLM v1 tối ưu hóa PP bằng cách cho phép xếp hàng nhiều batch/request chạy gối đầu liên tục (Multi-step/Micro-batching execution), giúp lấp đầy các khoảng thời gian trống này.

### 3.2. Quản lý KV Cache trong PP:
Một câu hỏi phổ biến: **Khi chạy Pipeline Parallelism, KV Cache của các layer có cần truyền qua lại giữa các GPU không?**

*   **Câu trả lời**: **Hoàn toàn không**.
*   **Cơ chế**:
    *   GPU 0 chỉ chứa trọng số của layer 1-16, do đó nó chỉ tạo và quản lý KV Cache của layer 1-16 trong bộ nhớ VRAM của chính nó.
    *   GPU 1 chỉ chứa trọng số của layer 17-32, do đó nó tự quản lý KV Cache của layer 17-32 trong VRAM của mình.
    *   Thông tin duy nhất cần truyền tải qua mạng giữa GPU 0 và GPU 1 ở mỗi bước chỉ là **Activation Tensor** đầu ra của layer 16 (kích thước cực kỳ nhỏ: $B \times 1 \times d$ đối với Decode). Do đó, PP cực kỳ tiết kiệm băng thông mạng so với TP.

---

## 4. Khảo sát Mã nguồn thực tế trong vLLM

Trong vLLM, các lớp tuyến tính song song được định nghĩa chi tiết tại tệp [vllm/model_executor/layers/linear.py](file:///Users/admin/TuanDung/repos/vllm/vllm/model_executor/layers/linear.py):

*   Lớp `ColumnParallelLinear` kế thừa từ `LinearBase`, thực hiện phép nhân ma trận và hỗ trợ gom/chia tensor.
*   Lớp `RowParallelLinear` thực hiện nhân ma trận và trực tiếp gọi hàm truyền thông All-Reduce:

```python
# Minh họa logic All-Reduce trong RowParallelLinear.forward
class RowParallelLinear(LinearBase):
    def forward(self, input_):
        # 1. Nhân ma trận cục bộ (Local Matrix Multiplication)
        out, bias = self.linear_method.apply(self, input_)
        
        # 2. Nếu TP Group > 1, kích hoạt All-Reduce đồng bộ
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
            
        return out + bias
```

---

## 💡 Tổng kết bài học

*   **Tensor Parallelism (TP)** chia mô hình theo chiều ngang (chia ma trận). `ColumnParallelLinear` chia cột và không cần truyền thông; `RowParallelLinear` chia dòng và yêu cầu **All-Reduce** để cộng gộp kết quả.
*   **Ring-based All-Reduce** của NCCL tối ưu hóa truyền thông bằng cách chia dữ liệu thành $N$ phần và truyền vòng tròn, giúp băng thông truyền trên mỗi GPU độc lập với số lượng node.
*   **Pipeline Parallelism (PP)** chia mô hình theo chiều dọc (chia layer). PP chỉ truyền **Activation Tensors** giữa các GPU ở biên layer, còn **KV Cache được giữ cố định** tại mỗi GPU tương ứng, giúp giảm tối đa overhead mạng liên node.
