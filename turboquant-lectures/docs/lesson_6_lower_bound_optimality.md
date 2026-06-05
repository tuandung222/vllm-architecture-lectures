---
sidebar_position: 7
sidebar_label: "Bài 6: Cận dưới & Tính tối ưu (~2.7)"
---

# Bài 6: Cận dưới lý thuyết & Tính tối ưu (~2.7)

Tuyên bố mạnh nhất của TurboQuant là *"near-optimal distortion rate"* — gần tối ưu. Nhưng "gần tối ưu" so với cái gì? Bài này làm rõ: paper **chứng minh một cận dưới (lower bound) thông tin** mà **mọi** vector quantizer đều phải tuân theo, rồi chỉ ra TurboQuant chỉ cách cận đó đúng một **hằng số $\approx 2.72$**, đồng đều ở mọi bit-width và mọi chiều.

---

## 1. Cận dưới nghĩa là gì và vì sao cần nó?

Trong khoa học thuật toán, một kết quả chỉ thực sự thuyết phục khi đi kèm **cận dưới**: chứng minh rằng **không tồn tại** thuật toán nào làm tốt hơn một ngưỡng nào đó. Không có cận dưới, ta không biết còn "dư địa" cải tiến hay không.

> [!IMPORTANT]
> Đóng góp lý thuyết cốt lõi của paper: *"a formal proof of the information-theoretic lower bounds on best achievable distortion rate by any vector quantizer."* Tức họ chứng minh: **với $b$ bit/tọa độ, KHÔNG quantizer nào (kể cả VQ tối ưu, kể cả biết trước dữ liệu) đạt méo nhỏ hơn $D_{\text{LB}}(b)$.** Rồi họ chỉ ra TurboQuant đạt $\le 2.72 \cdot D_{\text{LB}}(b)$.

---

## 2. Cận dưới đến từ Rate-Distortion của Shannon

Với nguồn mà mỗi tọa độ có phương sai $\sigma^2$, lý thuyết rate-distortion (Bài 0) cho cận dưới:

$$D_{\text{LB}}(b) = \sigma^2\, 2^{-2b}.$$

Đây là méo **không thể vượt qua** dù dùng codebook tối ưu trong $\mathbb R^d$. Mọi bộ nén $b$ bit/tọa độ đều nằm **phía trên** đường này.

---

## 3. Khoảng cách của TurboQuant: hằng số $\frac{\sqrt3\pi}{2}$

Ở Bài 3 ta đã tính méo thực tế của TurboQuant (rotation + scalar quantizer tối ưu):

$$D_{\text{TQ}}(b) = \frac{\sqrt 3\,\pi}{2}\,\sigma^2\, 2^{-2b}.$$

Vậy tỷ số:

$$\boxed{\;\frac{D_{\text{TQ}}(b)}{D_{\text{LB}}(b)} = \frac{\sqrt 3\,\pi}{2} \approx 2.72\;}$$

Điều phi thường: tỷ số này **là hằng số** — **không phụ thuộc $b$, không phụ thuộc $d$**. Dù bạn nén ở 2 bit, 4 bit hay 8 bit; dù vector 64 chiều hay 4096 chiều — TurboQuant luôn cách tối ưu **đúng ~2.72 lần**. Đó là ý nghĩa chặt chẽ của *"near-optimal across all bit-widths and dimensions"*.

```text
  Distortion (log scale)
   │
   │  ╲  ╲                      ── = cận dưới Shannon  D_LB = σ² 2^(−2b)
   │   ╲  ╲                     ┄┄ = TurboQuant       D_TQ = 2.72·D_LB
   │  ┄ ╲┄ ╲                    khoảng cách dọc luôn = log(2.72) ≈ const
   │     ╲  ╲┄
   │      ╲  ╲ ┄
   │       ╲  ╲  ┄
   └────────────────────────►  b (bit/tọa độ)
         hai đường SONG SONG trên thang log
```

---

## 4. Hằng số 2.72 đến từ đâu? (Diễn giải hình học)

Như Bài 0 & 3 đã hé lộ, khoảng cách này **không phải** do TurboQuant "dở", mà là **giá nội tại của việc dùng scalar quantizer** thay vì lattice nhiều chiều. Phân rã:

* **Shape gain** (khai thác tương quan + hình dạng): TurboQuant **không mất** phần này, vì random rotation đã làm các tọa độ i.i.d. Gauss — không còn gì để khai thác.
* **Space-filling gain**: ô Voronoi của scalar quantizer là **hình hộp** (siêu lập phương), trong khi ô tối ưu nhiều chiều gần **hình cầu**. Hình cầu "đóng gói" không gian hiệu quả hơn. Khoảng cách hiệu quả này, với nguồn Gauss, đúng bằng:

$$\frac{\sqrt 3\,\pi}{2} \approx 2.72 \quad (\approx 4.35\text{ dB}).$$

> [!NOTE]
> Đây là một hằng số **cơ bản của lý thuyết thông tin**, không phải tham số tinh chỉnh được. Để vượt qua nó, bắt buộc phải dùng **vector quantizer/lattice nhiều chiều** thật sự (như trellis-coded quantization) — vốn đắt và/hoặc data-dependent. TurboQuant chọn **hy sinh đúng 2.72×** để đổi lấy **tính online, data-oblivious, chi phí $O(d\log d)$**. Đó là một sự đánh đổi cực kỳ đáng giá cho serving.

---

## 5. Vì sao "hằng số không đổi" lại quan trọng trong thực tế?

Nhiều phương pháp lượng hóa "tốt ở bit cao nhưng sụp ở bit thấp" (hoặc ngược lại) vì khoảng cách tới tối ưu **giãn ra** ở vùng bit nào đó. TurboQuant thì **đều đặn**:

* **Dự đoán được**: biết méo ở 4 bit, suy ra ngay méo ở 2.5 bit (quy luật 6 dB/bit + hằng số). Kỹ sư serving có thể **chọn bit-width theo ngân sách VRAM** một cách định lượng.
* **Bền ở bit thấp**: chính vì không sụp đổ, TurboQuant mới đạt được **2.5 bit** mà chỉ "suy giảm biên" — vùng mà nhiều phương pháp khác đã hỏng.
* **Bền theo chiều**: dùng được cho cả $d_{\text{head}}=64,128$ (KV cache) lẫn $d=768,1536$ (embedding/vector DB) — Bài 7.

---

## 6. Tổng kết Bài 6

* Paper chứng minh **cận dưới thông tin** $D_{\text{LB}}(b)=\sigma^2 2^{-2b}$ mà **mọi** quantizer phải tuân theo.
* TurboQuant đạt $D_{\text{TQ}}(b) = \frac{\sqrt3\pi}{2}\sigma^2 2^{-2b}$, tức cách tối ưu **đúng hằng số $\approx 2.72$** — **không đổi theo bit-width và chiều**.
* Hằng số này là **space-filling gain** mất đi khi dùng scalar thay vì lattice — một giá cơ bản của lý thuyết thông tin, đổi lấy tính **online & data-oblivious**.
* Tính "hằng số không đổi" làm méo **dự đoán được** và **bền ở bit thấp** — yếu tố quyết định để dùng thực tế.

👉 Bài tiếp theo: **[Bài 7 — Ứng dụng Nearest Neighbor Search & Vector DB](./lesson_7_nearest_neighbor_search.md)**.
