---
sidebar_position: 11.5
sidebar_label: "Bài 7.3: Multi-LoRA Serving & SGMV/Punica"
---

# Bài 7.3: Kiến trúc Multi-LoRA Serving & Phép toán SGMV/Punica Kernels

Trong môi trường triển khai thực tế, doanh nghiệp thường muốn tùy biến mô hình nền (Base Model, ví dụ Llama 3 8B) cho nhiều tác vụ khác nhau: một adapter LoRA cho bộ phận chăm sóc khách hàng, một adapter cho phân tích tài chính, một adapter cho dịch thuật. 

Nếu chạy mỗi adapter trên một instance GPU độc lập, chi phí phần cứng sẽ cực kỳ đắt đỏ. Ngược lại, nếu nạp/hủy adapter liên tục khi có request, băng thông PCIe sẽ làm chậm hệ thống. vLLM giải quyết bài toán này qua cơ chế **Multi-LoRA Serving** cho phép phục vụ đồng thời hàng ngàn LoRA adapters trên cùng một GPU vật lý.

Bài học này sẽ phân tích chi tiết cơ chế quản lý LoRA Cache và các custom kernels nổi tiếng: **BGMV** và **SGMV**.

---

## 1. Bản chất Toán học của Multi-LoRA Serving

Nhắc lại phương pháp **LoRA (Low-Rank Adaptation)**: Thay vì cập nhật toàn bộ ma trận trọng số gốc $W_0 \in \mathbb{R}^{d \times k}$ (vốn có dung lượng rất lớn), LoRA giả định cập nhật trọng số $\Delta W$ có hạng thấp (low rank $r \ll d, k$), phân rã thành tích của hai ma trận nhỏ $A \in \mathbb{R}^{d \times r}$ và $B \in \mathbb{R}^{r \times k}$:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (A \cdot B)$$

Với vector kích hoạt đầu vào $x \in \mathbb{R}^{1 \times d}$, phép tính forward của lớp tuyến tính tích hợp LoRA là:

$$y = x \cdot W_0 + \frac{\alpha}{r} (x \cdot A) \cdot B$$

*   **Tính toán song song**: Trong phép tính này, phần nhân với ma trận gốc $x \cdot W_0$ là hoàn toàn giống nhau cho tất cả các request trong batch. Điểm khác biệt duy nhất là các ma trận nhỏ $A$ và $B$ tương ứng với từng adapter cụ thể của từng request.

---

## 2. Thách thức Gom lô (Batching Challenge) & custom GPU Kernels

Khi thực hiện gom lô động (Continuous Batching), chúng ta có một batch gồm các request trỏ tới các LoRA adapters khác nhau:
*   Token 1 dùng LoRA Adapter 3.
*   Token 2 dùng LoRA Adapter 12.
*   Token 3 không dùng LoRA (chỉ dùng Base Model).

Trong các thư viện PyTorch thông thường, phép nhân ma trận theo lô (`torch.bmm`) yêu cầu các ma trận trong lô phải có chung hình dáng (shape). Việc chạy tuần tự từng LoRA cho từng token sẽ phá vỡ hoàn toàn sức mạnh tính toán song song của GPU.

Để giải quyết vấn đề này, vLLM sử dụng các GPU kernels tối ưu hóa cực cao được kế thừa từ dự án **Punica**: **BGMV** và **SGMV**.

```
              ┌───────────────────────────────────────────────┐
              │ Batch Tokens X = [ x0, x1, x2, x3, x4, x5 ]   │
              └───────────────────────┬───────────────────────┘
                                      ▼
                       [ Nhân Base Model chung: X * W0 ]
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼ (Nhánh LoRA 1)                      ▼ (Nhánh LoRA 2)
           Tokens: [ x0, x1, x4 ]                Tokens: [ x2, x3, x5 ]
           LoRA:   Adapter A1, B1                LoRA:   Adapter A2, B2
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼ (Gọi BGMV / SGMV)
                    [ Gom nhóm và cộng tổng kết quả ]
                                      │
                                      ▼
                Output Y = X * W0 + Delta_W (X_i * A_Li * B_Li)
```

### 2.1. BGMV (Batch Gather Matrix-Vector multiplication)
*   **Mục đích**: Dùng cho pha **Decode** (bước sinh token).
*   **Đặc điểm**: Mỗi request trong batch chỉ có đúng $1$ token đầu vào ($N = 1$). Phép toán tương ứng là nhân ma trận với vector (Matrix-Vector).
*   **Cơ chế**: Kernel BGMV nhận vào một danh sách các con trỏ trỏ tới địa chỉ vật lý của các ma trận $A_{L_i}$ và $B_{L_i}$ trong bộ nhớ GPU dựa trên chỉ số adapter của từng request. Kernel tự động thực hiện phép nhân ma trận-vector cho từng token với cặp adapter tương ứng của nó một cách đồng thời trong một chu kỳ chạy Grid GPU, không tạo ra thời gian rảnh rỗi.

### 2.2. SGMV (Segmented Gather Matrix-Vector multiplication)
*   **Mục đích**: Dùng cho pha **Prefill** (xử lý prompt) hoặc các bước chạy sinh nhiều token đồng thời (Speculative Decoding).
*   **Đặc điểm**: Trong prefill, một request có thể chứa một phân đoạn (Segment) gồm nhiều tokens liên tiếp (ví dụ: request 1 có segment prompt dài 128 tokens đầu vào cùng sử dụng chung 1 adapter).
*   **Cơ chế**: Kernel SGMV chia nhỏ batch thành các phân đoạn (segments) có độ dài khác nhau. Nó thực hiện gom nhóm (Gather) các tokens trong cùng một phân đoạn để thực hiện nhân ma trận-ma trận (GEMM) với cặp ma trận $A$ và $B$ tương ứng của phân đoạn đó, đảm bảo hiệu suất tính toán tối ưu ngay cả khi độ dài prompt của các request trong batch chênh lệch lớn.

---

## 3. Hệ thống quản lý Bộ nhớ đệm LoRA (LoRA Cache)

Do VRAM GPU giới hạn, vLLM không thể nạp tất cả các adapter LoRA có sẵn trên ổ đĩa vào GPU. Thay vào đó, vLLM quản lý trọng số LoRA thông qua cơ chế **LoRA Cache**:

### 3.1. Phân cấp Bộ nhớ (Memory Hierarchy):
1.  **Host Cache (CPU RAM)**: Nơi lưu trữ tất cả các adapter LoRA sẵn có dưới định dạng PEFT (dung lượng có thể lên tới hàng chục Gigabytes).
2.  **Device Cache (GPU VRAM)**: Vùng nhớ đệm cố định trên GPU được cấp phát sẵn để chứa các ma trận LoRA đang hoạt động (Active Adapters).

### 3.2. Lập lịch và Swap (LoRA Scheduling):
*   Khi có request mới yêu cầu LoRA Adapter $X$:
    *   **Cache Hit**: Nếu Adapter $X$ đã nằm trong Device Cache, bộ lập lịch sẽ chèn request vào batch ngay lập tức.
    *   **Cache Miss**: Nếu Adapter $X$ chưa có trên GPU, bộ lập lịch sẽ yêu cầu lớp quản lý sao chép (Swap-in) trọng số của Adapter $X$ từ RAM CPU sang GPU VRAM qua khe PCIe.
*   **Trục xuất (Eviction)**: Nếu Device Cache trên GPU đầy, vLLM sử dụng thuật toán **LRU (Least Recently Used)** để trục xuất adapter lâu ngày không được sử dụng về CPU RAM để nhường chỗ cho adapter mới.

---

## 4. Khảo sát Mã nguồn thực tế trong vLLM

Toàn bộ hệ thống quản lý Multi-LoRA nằm trong thư mục [vllm/lora](file:///Users/admin/TuanDung/repos/vllm/vllm/lora):

*   [vllm/lora/model_manager.py](file:///Users/admin/TuanDung/repos/vllm/vllm/lora/model_manager.py): Quản lý vòng đời nạp, tráo đổi (swap) và đăng ký chỉ số của các adapter.
*   [vllm/lora/layers](file:///Users/admin/TuanDung/repos/vllm/vllm/lora/layers): Định nghĩa các lớp wrapper như `ColumnParallelLinearWithLoRA` để bọc các lớp tuyến tính nguyên bản, chèn thêm nhánh rẽ tính toán LoRA.
*   [vllm/lora/punica_wrapper](file:///Users/admin/TuanDung/repos/vllm/vllm/lora/punica_wrapper): Nơi vLLM gọi các custom kernels viết bằng CUDA/C++:

```python
# Minh họa việc gọi kernel BGMV/SGMV trong punica_wrapper
def apply_lora(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
):
    if x.size(0) == indices.size(0):
        # Decode phase: Mỗi token tương ứng 1 index -> Sử dụng BGMV
        bgmv(output, x, lora_a, lora_b, indices)
    else:
        # Prefill phase: Tokens phân đoạn -> Sử dụng SGMV
        sgmv(output, x, lora_a, lora_b, indices)
```

---

## 💡 Tổng kết bài học

*   **Multi-LoRA Serving** giải quyết bài toán kinh tế phục vụ hàng ngàn adapter trên cùng một instance GPU bằng cách chia sẻ trọng số Base Model.
*   **BGMV** tối ưu hóa phép nhân ma trận-vector cho pha Decode, cho phép chạy song song các token có adapter khác nhau trong 1 chu kỳ GPU.
*   **SGMV** tối ưu hóa phép nhân ma trận-ma trận cho pha Prefill bằng cách xử lý song song các phân đoạn token có độ dài biến thiên sử dụng chung adapter.
*   Hệ thống **LoRA Cache** điều phối việc nạp tráo đổi (Swap-in/Swap-out) adapter giữa RAM CPU và GPU VRAM qua PCIe, đảm bảo tối thiểu hóa thời gian chờ (latency) khi gặp cache miss.
