---
sidebar_position: 10
sidebar_label: "Bài 7.1: Chi tiết Lượng hóa – AWQ vs Activation"
---

# Bài 7.1: Chi tiết Lượng hóa – Weight-Only (AWQ) vs Activation Quantization

Lượng hóa (Quantization) là một trong những vũ khí tối thượng của AI Serving Engineer để nén mô hình và tăng tốc độ suy luận. Tuy nhiên, việc áp dụng lượng hóa không đơn thuần là làm tròn số. Chúng ta cần hiểu rõ sự khác biệt bản chất giữa **Weight-Only Quantization (như AWQ)** và **Activation Quantization (như W8A8 hoặc FP8)** để chọn giải pháp phù hợp nhất cho hệ thống phục vụ của mình.

---

## 1. Tại sao cần Lượng hóa trong LLM Serving?

Trong quá trình phục vụ mô hình ngôn ngữ lớn (LLM Serving), lượng hóa giải quyết hai điểm nghẽn bộ nhớ nghiêm trọng:
1. **Dung lượng Trọng số (Model Weights)**: Một mô hình 70B ở định dạng FP16 chiếm 140GB VRAM, bắt buộc phải dùng ít nhất 2 GPU 80GB để chạy. Lượng hóa xuống INT4 giúp nén dung lượng trọng số xuống còn 35GB, có thể chạy mượt mà chỉ trên 1 GPU đơn lẻ giá rẻ hơn.
2. **Băng thông nạp dữ liệu (HBM Bandwidth)**: Như phân tích ở Bài 0.1, pha Decode bị nghẽn do tốc độ đọc trọng số từ HBM lên SRAM. Nén trọng số giúp giảm lượng byte cần truyền tải, trực tiếp tăng tốc độ sinh token.

---

## 2. Weight-Only Quantization (như AWQ, GPTQ)

### 2.1. Nguyên lý hoạt động:
* **Đặc điểm**: Chỉ lượng hóa các trọng số tĩnh (weights) của mô hình từ FP16 xuống INT8 (1 Byte) hoặc INT4 (0.5 Byte). Các giá trị kích hoạt trung gian (activations) sinh ra trong quá trình tính toán vẫn được giữ nguyên ở định dạng dấu phẩy động độ chính xác cao (FP16/BF16).
* **Cơ chế thực thi trên GPU (On-the-fly Dequantization)**:
  1. Trọng số mô hình ở định dạng INT4 được nạp từ bộ nhớ HBM lên bộ nhớ đệm Shared Memory (SRAM) của SM. Nhờ kích thước nhỏ bằng 1/4 so với FP16, thời gian truyền tải qua bus bộ nhớ giảm đi 4 lần.
  2. Ngay trên SRAM, trước khi thực hiện phép nhân ma trận, một bộ chuyển đổi số chuyên dụng sẽ giải nén (dequantize) các trọng số INT4 này ngược trở lại thành FP16.
  3. Phép nhân ma trận-vector (GEMV) thực tế vẫn được thực hiện giữa hai tensor FP16 trên Tensor Cores.

```
HBM (RAM GPU)                   SRAM (Bộ đệm SM)              Tensor Cores (ALU)
┌──────────────┐  PCIe/Bus VRAM  ┌──────────────┐  Dequantize  ┌───────────────┐
│ Weights INT4 │ --------------> │ Weights INT4 │ -----------> │ Weights FP16  │
└──────────────┘  (Nhanh gấp 4x)  └──────────────┘  (Trên SRAM) └───────┬───────┘
                                                                        │  x (Phép nhân FP16)
                                                                        │
                                                               ┌────────▼───────┐
                                                               │ Activations    │
                                                               │    (FP16)      │
                                                               └────────────────┘
```

> [!TIP]
> **Tại sao Weight-Only cực kỳ hiệu quả cho pha Decode?**
> Vì pha Decode là **Memory-bound** (tốc độ bị giới hạn bởi thời gian đọc HBM). Việc tốn thêm một ít năng lực tính toán trên SRAM để giải nén INT4 -> FP16 là vô cùng nhỏ so với thời gian khổng lồ tiết kiệm được khi không phải nạp weights FP16 cồng kềnh qua bus VRAM.

### 2.2. Giải thuật AWQ (Activation-aware Weight Quantization) là gì?
Khi lượng hóa trọng số xuống INT4 một cách thô bạo (ví dụ làm tròn tuyến tính), độ chính xác của mô hình thường bị suy giảm rất nặng (hiện tượng perplexity tăng vọt, mô hình nói nhảm). AWQ giải quyết vấn đề này bằng một phát hiện quan trọng:

> **Phát hiện của AWQ**: Trọng số mô hình không có vai trò quan trọng như nhau. Bằng cách quan sát phân phối của các giá trị kích hoạt (Activations), ta thấy có một lượng nhỏ kênh kích hoạt (channels) có giá trị cực lớn (Outliers). Các trọng số tương ứng với các kênh kích hoạt lớn này (gọi là **Salient Weights**) đóng vai trò quyết định đến độ chính xác của mô hình.

```
Các kênh Kích hoạt (Activations):
[ 0.12  |  0.05  |  24.50 (Outlier!) |  -0.08 ]
                    |
                    v
Các Trọng số tương ứng (Weights):
[  w_0  |   w_1  |       w_2         |   w_3  ]
                         |
                         v
                Bảo vệ đặc biệt (AWQ):
      Không lượng hóa thô w_2, giữ nguyên tỷ lệ scale
```

* **Thuật toán AWQ**:
  1. AWQ chạy thử một lượng nhỏ dữ liệu huấn luyện để đo biên độ kích hoạt của từng kênh.
  2. Xác định các trọng số quan trọng (salient weights) tương ứng với các kênh có kích hoạt lớn.
  3. Thay vì lượng hóa toàn bộ, AWQ áp dụng một hệ số tỷ lệ (scaling factor) để bảo vệ các salient weights này khỏi bị sai số lượng hóa, chỉ lượng hóa thô các weights ít quan trọng hơn.
  4. Kết quả: Mô hình nén INT4 của AWQ giữ được độ chính xác gần như tương đương 99% so với mô hình FP16 gốc, tốt hơn nhiều so với phương pháp GPTQ truyền thống.

---

## 3. Activation Quantization (Lượng hóa cả Kích hoạt - W8A8 / FP8)

Khác với Weight-Only, các phương pháp lượng hóa toàn phần (như W8A8 - Weight 8bit, Activation 8bit hoặc định dạng FP8 mới của H100) lượng hóa **cả trọng số lẫn giá trị kích hoạt** xuống dạng 8-bit.

### 3.1. Tại sao lượng hóa kích hoạt lại rất khó?
Weights là các hằng số tĩnh, ta có thể phân tích và scale trước khi serving. Ngược lại, Activations biến đổi động theo từng dữ liệu đầu vào của người dùng.
Đặc biệt, trong các LLM có kích thước lớn (từ 6.7B trở lên), activations xuất hiện các giá trị cực đại bất thường (Outliers, ví dụ gấp 100 lần giá trị trung bình) nằm rải rác ở một số chiều nhất định. Nếu lượng hóa tuyến tính, các giá trị thường sẽ bị nén về 0, làm mất hoàn toàn thông tin đặc trưng của mô hình. Các giải thuật như SmoothQuant phải sinh ra các phép biến đổi toán học phức tạp để chuyển độ khó lượng hóa từ activations sang weights trước khi thực hiện.

### 3.2. Lợi ích vượt trội của Activation Quantization:
Mặc dù khó hiện thực, lượng hóa kích hoạt mang lại hai giá trị to lớn cho hiệu năng hệ thống:

1. **Thực thi phép tính trực tiếp trên Tensor Cores (Compute Speedup)**:
   * Do cả Weights và Activations đều ở dạng 8-bit (INT8 hoặc FP8), GPU không cần thực hiện bước giải nén (dequantize) ngược về FP16 trên SRAM.
   * GPU sẽ sử dụng trực tiếp các chỉ dẫn phần cứng INT8 hoặc FP8 Tensor Core để nhân ma trận trực tiếp.
   * **Kết quả**: Tốc độ tính toán của Tensor Cores tăng gấp đôi so với FP16 (ví dụ: H100 đạt 1000 TFLOPS FP16 nhưng đạt tới 2000 TFLOPS với FP8). Điều này cực kỳ có lợi cho pha **Prefill** (Compute-bound) hoặc khi serving với Batch size cực lớn khiến pha Decode chuyển dịch sang Compute-bound.

2. **Nén bộ nhớ động KV Cache (FP8 KV Cache)**:
   * Lượng hóa kích hoạt cho phép lưu trữ KV Cache dưới định dạng FP8 thay vì FP16.
   * Như tính toán ở Bài 1, KV Cache chiếm phần lớn không gian VRAM khi chạy batch lớn. Giảm KV Cache xuống FP8 giúp **tiết kiệm ngay 50% dung lượng VRAM** của bộ nhớ đệm động, cho phép phục vụ batch size lớn gấp đôi và hỗ trợ chiều dài chuỗi dài hơn nhiều trên cùng một phần cứng.

---

## 💡 Bảng so sánh tổng hợp (Góc nhìn Serving Engineer)

| Tiêu chí | Weight-Only Quantization (AWQ/INT4) | Weight-Activation Quantization (W8A8 / FP8) |
| :--- | :--- | :--- |
| **Đối tượng nén** | Chỉ nén Trọng số (Weights) | Nén cả Trọng số và Kích hoạt (và KV Cache) |
| **Phép toán Tensor Core** | Thực thi dạng FP16 (phải dequantize trên SRAM) | Thực thi trực tiếp dạng INT8 / FP8 |
| **Bản chất tối ưu** | Tối ưu Băng thông đọc HBM (Memory-bound) | Tối ưu cả Băng thông HBM và Tốc độ tính toán (Compute-bound) |
| **Độ chính xác** | Rất cao (nhờ thuật toán AWQ bảo vệ salient weights) | Dễ bị suy giảm hơn (đòi hỏi xử lý outlier activations) |
| **KV Cache Size** | Không đổi (vẫn là FP16/BF16) | Giảm 50% (lưu trữ dạng FP8/INT8) |
| **Phù hợp nhất cho** | Serving mô hình lớn trên ít GPU, pha Decode với Batch size nhỏ/vừa. | Hệ thống serving tải trọng cực cao, batch size khổng lồ, xử lý ngữ cảnh cực dài. |
