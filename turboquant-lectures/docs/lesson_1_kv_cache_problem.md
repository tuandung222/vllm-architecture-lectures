---
sidebar_position: 2
sidebar_label: "Bài 1: Bài toán KV Cache & Data-Oblivious"
---

# Bài 1: Bài toán nén KV Cache & yêu cầu Data-Oblivious

Bài 0 cho ta nền tảng lý thuyết. Bài này giải thích **vì sao bài toán mà TurboQuant giải lại quan trọng đến vậy** trong thực tế LLM serving, và **đâu là ràng buộc khắc nghiệt** (online, data-oblivious) khiến hầu hết phương pháp VQ truyền thống thất bại.

---

## 1. KV Cache: điểm nghẽn VRAM của LLM Serving

Khi một LLM sinh văn bản tự hồi quy (autoregressive), để tránh tính lại Attention cho toàn bộ lịch sử ở mỗi bước, nó **lưu lại** vector Key và Value của mọi token đã xử lý. Tập hợp này gọi là **KV Cache**.

Dung lượng KV Cache cho một request:

$$\text{Size}_{\text{KV}} = 2 \times L \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes}$$

trong đó hệ số $2$ là cho cả Key lẫn Value, $L$ là độ dài chuỗi.

> **Ví dụ Llama-3-8B** ($n_{\text{layers}}=32$, $n_{\text{kv\_heads}}=8$, $d_{\text{head}}=128$, FP16 = 2 byte):
> Mỗi token tốn $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ byte $= 128$ KB.
> Một chuỗi $L = 32{,}000$ token → **~4 GB chỉ riêng KV Cache cho một request**.

Hệ quả: với batch lớn và ngữ cảnh dài, KV Cache **chiếm phần lớn VRAM** và là thứ giới hạn số request đồng thời (throughput). Nén KV Cache xuống $b$ bit/kênh trực tiếp:
* **Tăng batch size** (phục vụ nhiều người dùng hơn trên cùng GPU).
* **Tăng độ dài ngữ cảnh** tối đa.
* **Giảm băng thông HBM** cần đọc ở pha decode (memory-bound) → tăng tốc độ sinh token.

---

## 2. Vì sao nén KV Cache lại KHÓ hơn nén Weights?

Đây là điểm mấu chốt phân biệt TurboQuant với các phương pháp như AWQ/GPTQ.

| Tiêu chí | Weight Quantization (AWQ, GPTQ) | KV Cache Quantization (TurboQuant) |
| :--- | :--- | :--- |
| **Đối tượng** | Trọng số **tĩnh**, biết trước khi serving | Vector K/V **sinh động** theo từng token, từng người dùng |
| **Thời điểm** | Offline, một lần (có thể chạy hàng giờ) | **Online**, ngay trong vòng lặp decode (microseconds) |
| **Calibration** | Có thể dùng dataset hiệu chỉnh | **Không có** dữ liệu tương lai để hiệu chỉnh |
| **Phân phối** | Cố định | **Trôi (distribution shift)** theo prompt, theo domain |

> [!IMPORTANT]
> KV Cache được sinh ra **từng vector một, ngay lúc chạy**. Bạn **không thể** dừng lại chạy K-means trên toàn bộ cache sau mỗi token. Bạn cũng **không biết trước** phân phối của các token tương lai. Vì vậy mọi phương pháp VQ **data-dependent** (cần học codebook) đều **không phù hợp**.

---

## 3. Ba yêu cầu thép cho một KV Cache Quantizer

Từ phân tích trên, một bộ lượng hóa KV Cache lý tưởng phải đạt:

1. **Data-Oblivious (calibration-free)**: thuật toán **không** phụ thuộc vào phân phối dữ liệu. Cùng một quy tắc lượng hóa áp cho mọi vector, bất kể nó đến từ prompt nào. → Miễn nhiễm distribution shift.
2. **Online & rẻ**: chi phí mã hóa mỗi vector phải cực thấp ($O(d \log d)$ trở xuống), không có bước training/indexing nặng.
3. **Near-optimal distortion**: dù đơn giản, méo vẫn phải gần cận Shannon — nếu không, chất lượng mô hình (perplexity, accuracy) sẽ sụp đổ ở bit thấp.

Trước TurboQuant, người ta thường phải **đánh đổi**: hoặc đơn giản nhưng méo lớn (uniform INT4/INT8 trực tiếp, hỏng ở bit thấp do outlier), hoặc chất lượng tốt nhưng data-dependent (VQ học codebook, không online được).

> 🎯 **TurboQuant phá vỡ sự đánh đổi này**: nó **data-oblivious VÀ near-optimal cùng lúc**. Bí quyết là chuyển độ khó từ "thiết kế codebook thông minh" sang "một phép xoay ngẫu nhiên" — vốn rẻ và không cần dữ liệu.

---

## 4. Vấn đề Outlier — kẻ thù số một của lượng hóa trực tiếp

Vì sao không thể chỉ làm tròn KV xuống INT4 một cách ngây thơ? Vì các vector activation trong LLM lớn chứa **outlier**: một số ít kênh (channel) có biên độ cực lớn (gấp hàng chục–trăm lần trung bình).

```
Một vector Key gốc (8 kênh minh họa):
[ 0.12 | -0.31 |  0.08 |  14.7 (OUTLIER!) | -0.05 |  0.22 | -0.18 |  0.09 ]
                                ▲
                  Kênh này "kéo căng" toàn bộ thang lượng hóa
```

Nếu chọn thang lượng hóa (scale) đủ lớn để bao trùm $14.7$, thì các giá trị nhỏ ($0.12, -0.05, \dots$) sẽ bị nén hết về cùng một mức → **mất sạch thông tin**. Đây là lý do INT4 ngây thơ làm perplexity tăng vọt.

> [!TIP]
> **Trực giác cốt lõi của TurboQuant**: Phép **xoay ngẫu nhiên** ở Bài 2 sẽ "**trải đều**" năng lượng của outlier ra khắp $d$ tọa độ. Sau khi xoay, không còn kênh nào đặc biệt lớn — mọi tọa độ có cùng phân phối (Beta, gần Gauss) đẹp đẽ và **không có outlier**. Đây cũng là ý tưởng chung với các phương pháp dùng **Hadamard transform** (QuIP#, QuaRot, SpinQuant), nhưng TurboQuant đẩy nó tới mức **gần tối ưu có chứng minh**.

---

## 5. Tổng kết Bài 1

* KV Cache là **điểm nghẽn VRAM** chính của LLM serving; nén nó tăng throughput, context length và tốc độ decode.
* Nén KV Cache khó hơn nén weights vì phải **online, không calibration, chịu distribution shift**.
* Một KV quantizer lý tưởng phải **data-oblivious + rẻ + near-optimal** — bộ ba mà các phương pháp cũ phải đánh đổi.
* **Outlier** phá hủy lượng hóa trực tiếp; chìa khóa là **xoay ngẫu nhiên để trải đều năng lượng**.

👉 Bài tiếp theo: **[Bài 2 — Random Rotation & Phân phối Beta](./lesson_2_random_rotation_beta.md)**, trụ cột đầu tiên và là "phép màu" toán học của TurboQuant.
