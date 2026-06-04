---
sidebar_position: 3.5
sidebar_label: "Bài 1.1: Chuyển dịch Decode sang Compute-bound"
---

# Bài 1.1: Phân tích Chuyển dịch Decode sang Compute-bound khi Batch Size cực lớn

Trong [Bài 0.1](./lesson_0_gpu_hardware.md) và [Bài 1](./lesson_1_memory_bottleneck.md), chúng ta đã làm quen với kết luận kinh điển: **Prefill là Compute-bound, còn Decode là Memory-bound**. 

Tuy nhiên, kết luận này chỉ đúng đối với các hệ thống serving phục vụ số lượng người dùng đồng thời nhỏ (Batch Size thấp). Trong môi trường serving công nghiệp quy mô lớn (như ChatGPT, Gemini) với hàng ngàn request đồng thời, khi **Batch Size ($B$) được đẩy lên cực lớn (ví dụ: $B \ge 512$ hoặc $1024$ trên GPU H100)**, pha Decode sẽ trải qua một sự chuyển dịch vật lý kỳ diệu: **Nó chuyển từ Memory-bound sang Compute-bound**.

Bài viết này sẽ đi sâu phân tích toán học và cơ chế phần cứng đằng sau sự chuyển dịch này.

---

## 1. Phân tích Toán học của Arithmetic Intensity theo Batch Size ($B$)

Để hiểu tại sao tăng Batch Size lại thay đổi giới hạn hiệu năng, chúng ta hãy dùng toán học để tính toán chính xác **Cường độ tính toán (Arithmetic Intensity - $I$)** của pha Decode dưới dạng một hàm số của $B$.

Giả sử chúng ta đang thực hiện bước Decode cho một lớp tuyến tính (Linear Layer) bất kỳ trong mô hình Transformer (ví dụ: QKV Projection hoặc MLP Layer) với:
* $B$: Batch Size (số lượng request/chuỗi đang sinh token song song).
* $d$: Kích thước ẩn của mô hình (Hidden Dimension - ví dụ: $d = 4096$ trên Llama-3-8B).
* Kiểu dữ liệu sử dụng: **FP16** hoặc **BF16** ($2$ Bytes cho mỗi tham số/kích hoạt).

### 1.1. Lượng Dữ liệu cần nạp từ HBM (Memory Access - $M$)
Trong pha Decode, ma trận trọng số $W$ có kích thước $(d \times d)$ cố định phải được nạp từ bộ nhớ VRAM (HBM) chậm chạp lên bộ nhớ đệm SRAM của chip GPU. 
* Lượng bộ nhớ để đọc trọng số $W$: 
  $$M_{\text{weights}} = d \times d \times 2 \text{ Bytes}$$
* Vector kích hoạt đầu vào $X$ có kích thước là $(B \times d)$ (mỗi request trong batch có 1 token đầu vào). Lượng bộ nhớ để đọc kích hoạt $X$:
  $$M_{\text{activations}} = B \times d \times 2 \text{ Bytes}$$

Tổng lượng dữ liệu $M$ cần truyền qua bus bộ nhớ (bỏ qua KV Cache để đơn giản hóa mô hình cốt lõi):
$$M(B) = M_{\text{weights}} + M_{\text{activations}} = 2d^2 + 2Bd \text{ Bytes}$$

### 1.2. Số Phép toán cần thực thi (Compute FLOPs - $C$)
Phép toán thực hiện là nhân ma trận kích hoạt $(B \times d)$ với ma trận trọng số $(d \times d)$ để cho ra ma trận kết quả $(B \times d)$.
* Đối với phép nhân ma trận thông thường, mỗi phần tử kết quả yêu cầu $1$ phép nhân và $1$ phép cộng (tương đương $2$ FLOPs).
* Tổng số phép toán cần thực hiện:
  $$C(B) = 2 \times B \times d^2 \text{ FLOPs}$$

### 1.3. Công thức Arithmetic Intensity ($I(B)$)
Cường độ tính toán $I(B)$ bằng tỷ số giữa số phép toán thực thi và lượng bộ nhớ truy cập:
$$I(B) = \frac{C(B)}{M(B)} = \frac{2 B d^2}{2 d^2 + 2 B d} = \frac{B \cdot d}{d + B} \text{ FLOP/Byte}$$

Chia cả tử và mẫu cho $d$, ta được:
$$I(B) = \frac{B}{1 + \frac{B}{d}} \text{ FLOP/Byte}$$

### 1.4. Phân tích giới hạn (Limit Analysis)
Chúng ta hãy xem xét các kịch bản của Batch Size $B$ so với chiều ẩn $d$ (thường $d \ge 4096$):

* **Trường hợp 1: Batch Size cực nhỏ ($B \to 1$)**
  Khi $B \ll d$ (ví dụ $B=1$):
  $$I(1) = \frac{1}{1 + \frac{1}{4096}} \approx 1 \text{ FLOP/Byte}$$
  Đây là kịch bản cực kỳ nghẽn bộ nhớ (**Memory-bound**). Chúng ta phải nạp $32\text{ MB}$ weights chỉ để thực hiện $32\text{ MFLOPs}$ tính toán.

* **Trường hợp 2: Batch Size tăng dần ($B \approx 128 \to 256$)**
  Khi $B$ tăng nhưng vẫn nhỏ hơn nhiều so với $d$, tỷ số $\frac{B}{d}$ rất nhỏ ($\approx 0.03 \to 0.06$). Ta có:
  $$I(B) \approx B \text{ FLOP/Byte}$$
  Lúc này, cường độ tính toán tăng tuyến tính theo Batch Size.

* **Trường hợp 3: Batch Size cực đại ($B \to \infty$)**
  Khi $B$ tiến tới vô cùng (hoặc vượt qua $d$):
  $$\lim_{B \to \infty} I(B) = d \text{ FLOP/Byte}$$
  Giới hạn tối đa của cường độ tính toán khi Batch Size cực lớn chính là chiều ẩn $d$ của mô hình (tương đương với pha Prefill).

---

## 2. Điểm Uốn của Roofline Model (The Knee Point)

Để biết khi nào pha Decode chuyển sang Compute-bound, ta phải so sánh $I(B)$ với **Knee Point** (Điểm Uốn) của GPU đang chạy.

```
Hiệu năng đạt được (TFLOPS)
  ^
  |                                        MÁI NHÀ: Giới hạn bởi Compute (Peak TFLOPS)
  |                                      ┌───────────────────────────────
  |                                     /
  |                                    / 
  |                                   /  
  |                                  /   
  |                                 /
  |                                /
  |                               /   <- Điểm Uốn (Knee Point)
  |                              /
  |  Decode B=1 (I=1)           / 
  |  [Memory-Bound]            /  Decode B=512 (I=455)
  |   x                       /    x [Compute-Bound]
  +---+----------------------+-----+---------------------------------> Arithmetic Intensity
      1                      150   455                                 (FLOPs/Byte)
```

### Ví dụ thực tế trên GPU NVIDIA A100 (Sử dụng Tensor Cores BF16):
* **Peak Compute Power (C):** $312 \text{ TFLOPS}$ (Tensor Cores).
* **Memory Bandwidth (HBM):** $2.0 \text{ TB/s}$.
* **Knee Point ($I_{\text{knee}}$):**
  $$I_{\text{knee}} = \frac{\text{Peak Compute}}{\text{Memory Bandwidth}} = \frac{312 \text{ TFLOPS}}{2.0 \text{ TB/s}} = 156 \text{ FLOP/Byte}$$

Để pha Decode đạt trạng thái Compute-bound trên GPU A100, ta cần:
$$I(B) \ge I_{\text{knee}} \implies \frac{B}{1 + \frac{B}{d}} \ge 156$$

Với mô hình Llama-3-8B ($d = 4096$):
$$\frac{B}{1 + \frac{B}{4096}} \ge 156 \implies B \ge \frac{156}{1 - \frac{156}{4096}} \approx 162$$

**Kết luận:** 
* Nếu **Batch Size $B < 162$**: Hệ thống nằm ở vùng **Memory-bound**.
* Nếu **Batch Size $B \ge 162$**: Hệ thống vượt qua điểm uốn và chuyển sang vùng **Compute-bound**! 
*(Đối với GPU H100 có sức mạnh tính toán $1000\text{ TFLOPS}$ và băng thông $3.35\text{ TB/s}$, Knee Point nằm ở khoảng $300\text{ FLOP/Byte}$, yêu cầu Batch Size $B \ge 320$ để đạt Compute-bound).*

---

## 3. Bản chất vật lý ở cấp độ Phần cứng: Amortization (Khấu hao)

Tại sao lại có sự chuyển dịch kỳ diệu này ở mức độ phần cứng? Câu trả lời nằm ở cơ chế **tái sử dụng dữ liệu trên thanh ghi (Registers) và bộ nhớ đệm SRAM**.

### Kịch bản 1: Batch Size $B = 1$
1. GPU nạp trọng số $W_{ij}$ từ HBM (VRAM) qua bus bộ nhớ lên SRAM, rồi đưa vào thanh ghi của SM.
2. SM thực hiện đúng $1$ phép toán: Nhân $W_{ij}$ với kích hoạt $x_j$ của người dùng duy nhất đó.
3. Hết phép toán, phần tử $W_{ij}$ bị đẩy ra khỏi SRAM để nạp trọng số của lớp tiếp theo.
4. **Kết quả:** Băng thông nạp $W$ bị lãng phí vì chỉ phục vụ đúng 1 phép tính duy nhất.

### Kịch bản 2: Batch Size $B = 512$
1. GPU nạp trọng số $W_{ij}$ từ HBM lên SRAM và nạp vào thanh ghi của SM **đúng 1 lần**.
2. Một khi $W_{ij}$ đã nằm trên thanh ghi của SM, nhân tính toán (Tensor Core) sẽ giữ nguyên giá trị này và thực hiện nhân song song nó với **512 kích hoạt khác nhau** đại diện cho 512 tokens của 512 request đang chạy đồng thời trong batch.
3. **Kết quả:** Chi phí băng thông nạp trọng số từ HBM đã được **khấu hao (amortized) 512 lần**. Số lượng Bytes đọc từ HBM không đổi, nhưng số phép tính FLOPs thực hiện tăng gấp 512 lần. 

```
               +----------------------------------------+
               |        Nạp Matrix W từ HBM (1 lần)     |
               +----------------------------------------+
                                    |
                                    v
                       +------------------------+
                       |    SRAM / Registers    |
                       +------------------------+
                         /     |     |   ...   \
                        v      v     v          v
                      Req 0  Req 1  Req 2     Req 511
                      [FLOP] [FLOP] [FLOP]    [FLOP]
```

Băng thông bộ nhớ HBM lúc này không còn là nút thắt cổ chai, bởi vì Tensor Cores đã bị "bão hòa" (saturated) do bận rộn thực thi khối lượng phép nhân ma trận khổng lồ.

---

## 4. Ý nghĩa Thực tiễn đối với Hệ thống Serving

Hiểu được cơ chế chuyển dịch này giúp các kỹ sư Serving đưa ra các quyết định tối ưu hóa quan trọng:

1. **Tại sao ta phải gom Batch Size càng lớn càng tốt?**
   Khi hệ thống đạt trạng thái Compute-bound, chúng ta đang khai thác tối đa công suất phần cứng GPU (TFLOPS đạt mức Peak). Chi phí cho mỗi token sinh ra (Cost per Token) giảm xuống mức thấp nhất có thể.

2. **Sự đánh đổi về Bộ nhớ (KV Cache Memory Trade-off)**
   Để chạy được Batch Size cực lớn (ví dụ $B = 512$), chúng ta cần một lượng VRAM khổng lồ để lưu trữ KV Cache của 512 requests này. 
   Nếu không có giải pháp quản lý bộ nhớ thông minh như **PagedAttention** (loại bỏ phân mảnh), GPU sẽ bị tràn bộ nhớ (Out-Of-Memory - OOM) trước khi chúng ta kịp đẩy Batch Size lên mức Compute-bound.

3. **Ảnh hưởng của Lượng hóa (Quantization)**
   Khi áp dụng lượng hóa trọng số (Weight-only Quantization như AWQ, GPTQ INT4), dung lượng weights giảm đi 4 lần. Điều này làm giảm lượng dữ liệu $M_{\text{weights}}$ cần nạp từ HBM đi 4 lần, giúp **Knee Point dịch chuyển sang bên trái**.
   * Nhờ đó, chúng ta có thể đạt trạng thái Compute-bound ở một **Batch Size nhỏ hơn nhiều** (ví dụ thay vì cần $B=162$ trên A100, ta chỉ cần $B \approx 40$ đối với mô hình lượng hóa INT4).

---

## 💡 Tóm tắt bài học

1. **Arithmetic Intensity** của pha Decode tăng tuyến tính theo Batch Size $B$ theo công thức $I(B) \approx \frac{B}{1 + B/d}$.
2. Khi Batch Size nhỏ, Decode bị **Memory-bound** do chi phí nạp weights khổng lồ từ HBM chỉ phục vụ số ít phép tính.
3. Khi Batch Size vượt qua Điểm Uốn (Knee Point) của GPU (ví dụ $B \ge 162$ trên A100), **Decode trở thành Compute-bound** nhờ cơ chế tái sử dụng và khấu hao weights trên chip.
4. Quản lý KV Cache tối ưu (PagedAttention) và Lượng hóa (Quantization) là hai vũ khí tối thượng giúp serving engine đạt được trạng thái Compute-bound hiệu quả trong thực tế.
