---
sidebar_position: 2
sidebar_label: "Bài 0.1: Cấu tạo phần cứng GPU & Prefill/Decode"
---

# Bài 0.1: Cấu tạo Phần cứng GPU & Bản chất Vật lý của Prefill/Decode

Để tối ưu hóa phần mềm Serving, ta phải hiểu rõ cơ cấu hoạt động của phần cứng thực thi bên dưới - GPU. Trong bài học này, chúng ta sẽ phân tích kiến trúc phần cứng của GPU (từ SMs, Tensor Cores đến phân cấp bộ nhớ HBM/SRAM) và dùng mô hình toán học **Roofline Model** để giải thích bản chất vật lý của hai pha **Prefill** (Compute-bound) và **Decode** (Memory-bound).

---

## 1. Kiến trúc phần cứng GPU: SMs, Tensor Cores và Phân cấp Bộ nhớ

Một GPU hiện đại (như NVIDIA A100 hoặc H100) không phải là một CPU nhiều lõi. Nó là một bộ xử lý song song khối lượng cực lớn hoạt động theo mô hình **SIMT (Single Instruction, Multiple Threads)**.

```
+-------------------------------------------------------------------------+
|                               GPU VRAM (HBM)                            |
|       Dung lượng lớn (40GB - 80GB) | Băng thông trung bình (1.5 - 3 TB/s)  |
+-------------------------------------------------------------------------+
                                     |  (PCIe / HBM Bus)
                                     v
+-------------------------------------------------------------------------+
|                              L2 Cache (SRAM)                            |
|                       Dung lượng nhỏ (tầm 40MB - 80MB)                  |
+-------------------------------------------------------------------------+
                                     |
                                     v
   +-------------------------------------------------------------------+
   |             Streaming Multiprocessors (SM) - Nhân GPU             |
   | - L1 Cache / Shared Memory (SRAM) - Siêu nhanh (cỡ KB/MB, >10TB/s)|
   | - Registers (Thanh ghi)                                          |
   | - CUDA Cores (Phép tính FP32/INT32 thông thường)                 |
   | - Tensor Cores (Phép nhân ma trận FP16/BF16/FP8 siêu tốc)        |
   +-------------------------------------------------------------------+
```

### 1.1. Streaming Multiprocessor (SM) - Đơn vị tính toán cốt lõi
Một GPU được cấu thành từ hàng trăm **SMs** (ví dụ: GPU H100 có 132 SMs). Mỗi SM là một vi xử lý độc lập có chứa:
* **CUDA Cores (ALUs)**: Các lõi tính toán logic số học cơ bản (FP32, FP64, INT32).
* **Tensor Cores**: Các khối mạch phần cứng chuyên dụng được thiết kế riêng cho phép nhân ma trận tích lũy (FMA - Fused Multiply-Add) cực nhanh: $D = A \times B + C$ trên các kiểu dữ liệu dấu phẩy động độ chính xác thấp (FP16, BF16, FP8, INT8). Tensor Cores chính là động cơ phản lực đằng sau sự bùng nổ của Deep Learning.
* **Registers (Thanh ghi)**: Bộ nhớ trong cực nhanh chứa các biến đang tính toán trực tiếp của thread.

### 1.2. Phân cấp Bộ nhớ của GPU (Memory Hierarchy)
Khoảng cách về tốc độ giữa bộ nhớ chứa dữ liệu và nhân tính toán trên GPU là rất lớn. Để tối ưu hóa, GPU sử dụng kiến trúc bộ nhớ phân cấp:

1. **HBM / VRAM (High Bandwidth Memory)**:
   * Bản chất: Bộ nhớ chính của GPU (DRAM xếp chồng 3D để tăng chiều rộng bus dữ liệu).
   * Đặc điểm: Dung lượng lớn (80GB trên H100) nhưng tốc độ nạp dữ liệu vào chip chậm nhất trong hệ thống phân cấp (băng thông $\approx 3.35 \text{ TB/s}$).
2. **SRAM (L2 Cache & L1/Shared Memory)**:
   * Bản chất: Bộ nhớ tĩnh nằm trực tiếp trên chip xử lý GPU.
   * **L2 Cache**: Bộ nhớ đệm dùng chung cho toàn bộ các SM (dung lượng cỡ 50MB-80MB).
   * **L1 Cache / Shared Memory**: Bộ nhớ đệm tốc độ cực cao nằm riêng trong từng SM (dung lượng cỡ 100KB-250KB mỗi SM). Băng thông truy cập Shared Memory cực kỳ khủng khiếp (lên tới $>10 \text{ TB/s}$).

---

## 2. Bản chất Vật lý của Prefill vs Decode

Tại sao pha Prefill lại bị nghẽn do tính toán (Compute-bound) còn pha Decode lại nghẽn do đọc bộ nhớ (Memory-bound)? Câu trả lời nằm ở khả năng **tái sử dụng dữ liệu trên SRAM**.

### 2.1. Pha Prefill: Phép nhân GEMM (General Matrix Multiply)
Trong pha Prefill, chúng ta nạp vào mô hình toàn bộ Prompt cùng một lúc (ví dụ $N = 1024$ tokens).
* **Phép toán thực thi**: Nhân ma trận kích hoạt kích thước ($N \times d$) với ma trận trọng số (weights) kích thước ($d \times d$). Đây là phép toán **GEMM**.

```
Kích hoạt (Prompt):            Trọng số (Weights):           Output (Hidden States):
  (1024 x 4096)                   (4096 x 4096)                  (1024 x 4096)
┌──────────────┐                 ┌──────────────┐               ┌──────────────┐
│  A (Prompt)  │        x        │  W (Weights) │       =       │   D (Output) │
└──────────────┘                 └──────────────┘               └──────────────┘
```

* **Vật lý truyền dữ liệu**:
  1. Mỗi block trọng số $W$ được nạp từ bộ nhớ HBM chậm chạp lên bộ nhớ Shared Memory (SRAM) tốc độ cao của SM **đúng 1 lần**.
  2. Một khi đã nằm trên SRAM, block trọng số $W$ này được **tái sử dụng $1024$ lần** để nhân với toàn bộ 1024 vector kích hoạt tương ứng với 1024 tokens trong Prompt.
* **Arithmetic Intensity**: Số phép tính FLOPs cực lớn so với số Byte đọc từ HBM. GPU dành phần lớn thời gian để tính toán trên Tensor Cores mà không phải chờ đợi nạp dữ liệu. Hệ thống đạt trạng thái **Compute-bound**.

### 2.2. Pha Decode: Phép nhân GEMV (General Matrix-Vector Multiply)
Trong pha Decode, ở mỗi bước lặp chúng ta chỉ nạp vào mô hình đúng **1 token** vừa sinh ra từ bước trước ($N = 1$).
* **Phép toán thực thi**: Nhân vector kích hoạt kích thước ($1 \times d$) với ma trận trọng số kích thước ($d \times d$). Đây là phép toán **GEMV**.

```
Kích hoạt (1 Token):           Trọng số (Weights):           Output:
   (1 x 4096)                     (4096 x 4096)                 (1 x 4096)
  ┌──────────┐                   ┌──────────────┐              ┌──────────┐
  │ A (1tok) │         x         │  W (Weights) │      =       │ D(Output)│
  └──────────┘                   └──────────────┘              └──────────┘
```

* **Vật lý truyền dữ liệu**:
  1. Trọng số mô hình $W$ vẫn có kích thước khổng lồ ($d \times d$). Chúng ta vẫn phải nạp toàn bộ ma trận $W$ này từ bộ nhớ HBM lên SRAM của SM.
  2. Một khi đã lên SRAM, mỗi tham số trong ma trận $W$ chỉ được **tái sử dụng đúng 1 lần** duy nhất để nhân với 1 phần tử của vector kích hoạt đơn lẻ của token đó, rồi ngay lập tức bị ghi đè/giải phóng để nhường chỗ nạp các trọng số của tầng tiếp theo.
* **Arithmetic Intensity**: Số phép tính FLOPs rất ít ($2 \times d^2$) trong khi số Byte đọc từ HBM cực kỳ lớn ($2 \times d^2$ bytes đối với FP16). Tỷ lệ FLOP/Byte $\approx 1$. 
* Nhân Tensor Cores của GPU chạy siêu nhanh và hoàn thành phép tính trong vài micro-giây, sau đó phải **ngồi chơi xơi nước** hàng trăm micro-giây để đợi HBM chuyển các trọng số tiếp theo lên qua bus bộ nhớ. Hệ thống ở trạng thái **Memory-bound**.

---

## 3. Biểu diễn bằng toán học: Mô hình Roofline Model

Để mô tả trực quan giới hạn hiệu năng của GPU, các kỹ sư hệ thống sử dụng **Roofline Model** (Mô hình mái nhà).

```
Hiệu năng đạt được (TFLOPS)
  ^
  |               MÁI NHÀ: Giới hạn bởi Compute (Peak TFLOPS)
  |             ┌───────────────────────────────────────────────
  |            /
  |           / 
  |          /  SƯỜN DỐC: Giới hạn bởi Memory Bandwidth (Băng thông HBM)
  |         /   Hiệu năng = Arithmetic Intensity * Bandwidth
  |        /
  |       /
  |      /   <- Điểm Uốn (Knee Point)
  |     /
  +----+-------------------------------------------------------> Arithmetic Intensity
       0                                                         (FLOPs/Byte)
```

Trục tung biểu diễn hiệu năng tính toán thực tế đạt được (TFLOPS). Trục hoành biểu diễn cường độ tính toán (Arithmetic Intensity - FLOPs/Byte).

* **Đường sườn dốc (Memory-Bound Region)**: Khi cường độ tính toán của chương trình thấp hơn điểm uốn (Knee Point). Hiệu năng hệ thống bị giới hạn hoàn toàn bởi tốc độ của bộ nhớ:
  $$\text{Performance (TFLOPS)} = \text{Arithmetic Intensity (FLOP/Byte)} \times \text{Memory Bandwidth (TB/s)}$$
  * Muốn tăng hiệu năng ở vùng này, ta chỉ có 2 cách: dùng phần cứng có băng thông bộ nhớ cao hơn (ví dụ nâng cấp từ PCIe sang HBM3) hoặc áp dụng thuật toán giảm lượng dữ liệu nạp từ bộ nhớ (như Quantization).
* **Đường mái ngang (Compute-Bound Region)**: Khi cường độ tính toán vượt qua điểm uốn. Lúc này bộ nhớ cung cấp đủ dữ liệu cho nhân GPU chạy hết công suất. Hiệu năng hệ thống đạt mức đỉnh (Peak Performance) của phần cứng.

### So sánh vị trí của Prefill và Decode trên biểu đồ:
* **Pha Prefill**: Nằm rất sâu ở vùng **Compute-Bound** (mái ngang).
* **Pha Decode (với Batch size nhỏ)**: Nằm rất thấp ở vùng **Memory-Bound** (sườn dốc).

> [!IMPORTANT]
> **Kết luận**:
> Mục tiêu tối thượng của các thư viện Serving như vLLM là làm thế nào để kéo pha Decode từ vùng Memory-bound lên gần vùng Compute-bound nhất có thể. 
> Giải pháp là tăng **Batch Size** (Continuous Batching) để gom nhiều vector kích hoạt của nhiều request lại, giúp tăng tỷ số tái sử dụng trọng số trên SRAM ở mỗi bước lặp, đồng thời áp dụng **Quantization** để giảm dung lượng weights nạp vào SRAM.
