---
sidebar_position: 4
sidebar_label: "Bài 3: MSE Scalar Quantizer tối ưu"
---

# Bài 3: Trụ cột 2 — MSE Scalar Quantizer tối ưu

Bài 2 đã biến vector $x$ thành $\tilde x = Rx$ với các tọa độ **i.i.d. xấp xỉ Gauss**. Giờ nhiệm vụ chỉ còn là: thiết kế **một** bộ lượng hóa vô hướng $q^\star$ tối ưu cho phân phối đó, rồi áp cho mọi tọa độ. Bài này trình bày cách thiết kế và — quan trọng nhất — **tính chính xác méo** của nó, từ đó suy ra công thức $D(b)$ then chốt.

---

## 1. Bộ lượng hóa vô hướng tối ưu: Lloyd–Max

Cho một biến ngẫu nhiên $u$ với mật độ $p(u)$ (ở đây $u = \sqrt d\,\tilde x_i \approx \mathcal N(0,1)$). Một bộ lượng hóa $b$ bit có $K = 2^b$ mức tái tạo $\{r_1,\dots,r_K\}$ và các biên quyết định $\{t_0,\dots,t_K\}$. Méo MSE:

$$D = \sum_{j=1}^{K} \int_{t_{j-1}}^{t_j} (u - r_j)^2\, p(u)\, du.$$

Cực tiểu hóa $D$ cho **điều kiện tối ưu Lloyd–Max**:

1. **Điều kiện trọng tâm (centroid)**: mỗi mức tái tạo là kỳ vọng có điều kiện của ô của nó
   $$r_j = \mathbb{E}[\,u \mid t_{j-1} \le u < t_j\,] = \frac{\int_{t_{j-1}}^{t_j} u\,p(u)\,du}{\int_{t_{j-1}}^{t_j} p(u)\,du}.$$
2. **Điều kiện lân cận gần nhất (nearest neighbor)**: biên nằm giữa hai mức tái tạo
   $$t_j = \frac{r_j + r_{j+1}}{2}.$$

Lặp xen kẽ hai điều kiện này (thuật toán Lloyd, tương đương K-means 1 chiều) hội tụ về quantizer **tối ưu MSE** cho phân phối $p$. Vì $p$ ở đây **cố định và biết trước** (chuẩn tắc), ta tính $q^\star$ **một lần offline** và nhúng cứng bảng vào code — không cần dữ liệu người dùng. Đây là điều giữ cho thuật toán **data-oblivious**.

> [!NOTE]
> Trong thực tế, người ta thường tách $\tilde x = \lVert x\rVert \cdot \hat u$ thành (a) **độ lớn (norm)** $\lVert x\rVert$ — lượng hóa riêng bằng vài bit, và (b) **hướng** $\hat u$ với các tọa độ phân phối đã biết. TurboQuant lượng hóa norm gần như không mất gì (norm là một số vô hướng duy nhất cho cả vector $d$ chiều).

---

## 2. Méo của quantizer tối ưu: lý thuyết high-rate

Để có **công thức đóng (closed-form)** cho méo, ta dùng **lý thuyết lượng hóa độ phân giải cao (high-rate quantization theory)**. Khi số mức $K=2^b$ đủ lớn, méo của bộ lượng hóa vô hướng tối ưu (companding) cho nguồn mật độ $p$ là:

$$\boxed{\;D_{\text{SQ}}(b) = \frac{1}{12}\,2^{-2b}\left(\int p(u)^{1/3}\,du\right)^{3}\;}$$

Đây là **công thức Panter–Dite / Bennett**. Số hạng $\frac{1}{12}$ là méo của lượng hóa đều (uniform) trên một ô độ rộng 1; phần còn lại là "hệ số hình dạng" của phân phối.

### Áp dụng cho nguồn Gauss $\mathcal N(0,\sigma^2)$

Tính tích phân $\int p^{1/3}$ cho mật độ Gauss cho ra $\left(\int p^{1/3}\right)^3 = 6\sqrt 3\,\pi\,\sigma^2$. Thay vào:

$$\boxed{\;D_{\text{SQ}}(b) = \frac{\sqrt 3\,\pi}{2}\,\sigma^2\, 2^{-2b} \;\approx\; 2.72\,\sigma^2\, 2^{-2b}\;}$$

So sánh với **cận Shannon** $D(b) = \sigma^2 2^{-2b}$ từ Bài 0:

$$\frac{D_{\text{SQ}}(b)}{D_{\text{Shannon}}(b)} = \frac{\sqrt 3\,\pi}{2} \approx 2.72 \qquad (\text{const, independent of } b).$$

> [!IMPORTANT]
> **Đây là con số 2.7 nổi tiếng của TurboQuant.** Bộ lượng hóa vô hướng tối ưu cho từng tọa độ Gauss đạt méo chỉ **lớn hơn cận tối ưu lý thuyết đúng một hằng số $\frac{\sqrt3\pi}{2}\approx 2.72$**, và hằng số này **giữ nguyên ở mọi bit-width $b$ và mọi chiều $d$**. Đó chính là ý nghĩa của cụm *"near-optimal distortion rate across all bit-widths and dimensions"* trong tiêu đề paper.

---

## 3. Vì sao "chia để trị" theo tọa độ lại gần tối ưu?

Một câu hỏi tự nhiên: lượng hóa độc lập từng tọa độ thường **kém** lượng hóa cả khối (Bài 0 nói VQ tốt hơn SQ). Vậy sao ở đây lại "gần tối ưu"?

Câu trả lời nằm ở chỗ **phép xoay đã làm hết việc khó**:

| Nguồn lợi ích của VQ | Sau random rotation? |
| :--- | :--- |
| **Shape gain** (khai thác tương quan + hình dạng phân phối) | ❌ Đã bị "tiêu" — các tọa độ giờ **i.i.d. Gauss**, không còn tương quan để khai thác. |
| **Space-filling gain** (ô Voronoi nhiều chiều lấp đầy tốt hơn hộp) | ✅ Vẫn còn — và đây chính là toàn bộ khoảng cách $2.72$. |

Nói cách khác: trong khi VQ tổng quát phải vật lộn với cả tương quan lẫn hình dạng phân phối (đắt, data-dependent), TurboQuant **dùng một phép xoay ngẫu nhiên rẻ tiền để xóa sạch phần khó đó**, chỉ để lại space-filling gain — một khoảng cách hằng số nhỏ mà ta vui vẻ chấp nhận để đổi lấy tính online & data-oblivious.

---

## 4. Diễn giải bit-width: bao nhiêu bit cho KV Cache?

Từ $D_{\text{SQ}}(b) \approx 2.72\,\sigma^2 2^{-2b}$, quy luật **6 dB/bit** vẫn đúng: mỗi bit thêm giảm méo $4\times$. Điều này khớp với kết quả thực nghiệm của paper trên KV Cache:

| Bit/kênh | Méo tương đối | Chất lượng mô hình (thực nghiệm paper) |
| :---: | :--- | :--- |
| **3.5 bit** | rất thấp | **Trung tính tuyệt đối** — gần như không phân biệt được với FP16 |
| **2.5 bit** | thấp | **Suy giảm biên (marginal)** — chấp nhận được cho hầu hết tác vụ |
| $\le 2$ bit | cao hơn | bắt đầu thấy ảnh hưởng, cần QJL (Bài 4) để giữ attention chính xác |

> [!TIP]
> Con số **3.5 bit cho chất lượng trung tính** rất ấn tượng: so với FP16 (16 bit) đó là **nén ~4.5×** KV Cache mà gần như không mất chất lượng — trực tiếp tăng batch size / context length lên tương ứng khi serving bằng vLLM (Bài 5).

---

## 5. Tổng kết Bài 3

* Thiết kế quantizer vô hướng tối ưu bằng **Lloyd–Max** (centroid + nearest-neighbor), tính **một lần offline** cho phân phối Gauss đã biết → data-oblivious.
* **Lý thuyết high-rate** cho công thức đóng: $D_{\text{SQ}}(b) = \frac{\sqrt3\pi}{2}\sigma^2 2^{-2b} \approx 2.72\,\sigma^2 2^{-2b}$.
* Khoảng cách tới cận Shannon là **hằng số $\approx 2.72$**, không đổi theo $b$ và $d$ — nguồn gốc của tuyên bố "near-optimal".
* Phép xoay đã "tiêu" shape gain, chỉ còn space-filling gain ($=$ hằng số $2.72$) là phần thiếu.
* Thực nghiệm: **3.5 bit trung tính, 2.5 bit suy giảm biên** cho KV Cache.

👉 Bài tiếp theo: **[Bài 4 — Inner Product & QJL Unbiased](./lesson_4_inner_product_qjl.md)**, xử lý vấn đề thiên lệch khi ước lượng attention scores.
