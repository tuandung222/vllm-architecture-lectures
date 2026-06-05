---
sidebar_position: 1
sidebar_label: "Bài 0: Nền tảng Vector Quantization"
---

# Bài 0: Nền tảng Vector Quantization & Rate-Distortion

Trước khi mổ xẻ TurboQuant, chúng ta cần một nền tảng vững chắc về **lượng hóa (quantization)** dưới góc nhìn của **lý thuyết mã hóa nguồn (source coding)** của Shannon. Bài này trả lời ba câu hỏi: (1) Méo (distortion) là gì và đo thế nào? (2) Giới hạn tốt nhất có thể đạt được (rate-distortion bound) là bao nhiêu? (3) Vì sao lượng hóa **vector** về nguyên tắc luôn tốt hơn lượng hóa **từng số (scalar)**?

---

## 1. Lượng hóa là một bài toán nén có mất mát (Lossy Compression)

Một **bộ lượng hóa (quantizer)** $Q$ ánh xạ một vector liên tục $x \in \mathbb{R}^d$ vào một tập hữu hạn các **điểm mã (codewords)** $\mathcal{C} = \{c_1, \dots, c_K\}$:

$$Q(x) = \arg\min_{c \in \mathcal{C}} \lVert x - c \rVert.$$

Nếu dùng $b$ bit cho mỗi tọa độ thì tổng ngân sách là $R = b \cdot d$ bit, tức $K = 2^{R} = 2^{bd}$ codewords. Vì $K$ hữu hạn còn $\mathbb{R}^d$ vô hạn, ta luôn mất thông tin → đây là **nén có mất mát**.

Có hai đại lượng đối nghịch:
* **Rate** $R$: số bit dùng để mô tả (càng nhỏ càng tốt cho bộ nhớ/băng thông).
* **Distortion** $D$: độ méo giữa $x$ và $\hat{x} = Q(x)$ (càng nhỏ càng tốt cho chất lượng).

Mục tiêu của mọi thuật toán quantization là đẩy điểm hoạt động $(R, D)$ càng gần **đường biên rate-distortion lý tưởng** càng tốt.

---

## 2. Hai loại Distortion: MSE vs Inner Product

TurboQuant đặc biệt ở chỗ nó tối ưu **cả hai** loại méo dưới đây — điều mà nhiều phương pháp trước bỏ sót.

### 2.1. Mean-Squared Error (MSE) Distortion

Đây là thước đo "hình học" quen thuộc — sai số tái tạo vector:

$$D_{\text{MSE}} = \mathbb{E}\big[\, \lVert x - \hat{x} \rVert_2^2 \,\big].$$

MSE quan trọng khi bạn cần **bản thân vector** chính xác (ví dụ tái tạo trọng số mô hình).

### 2.2. Inner Product Distortion

Trong rất nhiều ứng dụng, thứ ta thực sự cần không phải là $\hat{x}$ mà là **tích vô hướng** giữa nó với một vector truy vấn $q$:

$$\langle q, x \rangle \;\approx\; \langle q, \hat{x} \rangle.$$

Đây chính xác là thứ xảy ra trong **Attention**: điểm số attention là $\langle q_{\text{query}}, k_{\text{key}} \rangle$, và đầu ra là tổ hợp $\sum_i a_i v_i$ — đều là tích vô hướng với các vector Key/Value đã nén. Tương tự, **Maximum Inner Product Search (MIPS)** trong vector database cũng đo bằng tích vô hướng.

> [!IMPORTANT]
> **Điểm mấu chốt mà TurboQuant phát hiện**: Một quantizer tối ưu cho MSE **chưa chắc** tốt cho inner product. Cụ thể, quantizer tối ưu MSE thường tạo ra **thiên lệch (bias)**: $\mathbb{E}[\langle q, \hat{x}\rangle] \neq \langle q, x\rangle$. Bias này tích lũy và làm hỏng ước lượng attention. Bài 4 sẽ giải quyết bằng QJL.

---

## 3. Đường biên Rate-Distortion: giới hạn tốt nhất có thể

Shannon chứng minh rằng với một nguồn ngẫu nhiên cho trước, tồn tại một **giới hạn lý thuyết** $D(R)$ mà **không** bộ mã hóa nào (dù thông minh đến đâu) vượt qua được. Với nguồn **Gauss** $\mathcal{N}(0, \sigma^2)$ và thước đo MSE, hàm rate-distortion là:

$$\boxed{D(R) = \sigma^2 \, 2^{-2R}}$$

trong đó $R$ là số bit trên mỗi chiều. Đây là công thức nền tảng mà chúng ta sẽ tham chiếu xuyên suốt:

* **Quy luật 6 dB/bit**: mỗi bit thêm vào giảm méo đi $4\times$ (tức $-6$ dB). Đây là "tỷ giá hối đoái" giữa bộ nhớ và chất lượng.
* Tại sao Gauss lại quan trọng? Vì — như Bài 2 sẽ chỉ ra — **sau khi xoay ngẫu nhiên**, mỗi tọa độ của vector trở nên xấp xỉ Gauss. Đây là lý do TurboQuant "ép" dữ liệu về dạng Gauss để áp dụng lý thuyết đẹp đẽ này.

---

## 4. Scalar Quantization vs Vector Quantization

### 4.1. Scalar Quantization (SQ)

Lượng hóa **độc lập từng tọa độ**: với mỗi $x_i$, làm tròn về điểm mã gần nhất trên trục số 1 chiều. Đơn giản, nhanh, dễ song song hóa. Đây là cách AWQ/INT8/FP8 hoạt động.

### 4.2. Vector Quantization (VQ)

Lượng hóa **cả khối $d$ chiều cùng lúc**, dùng một codebook gồm các điểm trong $\mathbb{R}^d$ (ví dụ K-means / Product Quantization). VQ về lý thuyết **luôn tốt hơn hoặc bằng** SQ vì nó khai thác được hai loại "lợi ích" mà SQ không có:

| Lợi ích | Bản chất | SQ có? | VQ có? |
| :--- | :--- | :---: | :---: |
| **Space-filling gain** | Ô Voronoi nhiều chiều (như hình cầu) lấp đầy không gian hiệu quả hơn hình hộp của lưới vuông. | ❌ | ✅ |
| **Shape/Correlation gain** | Khai thác tương quan giữa các tọa độ và hình dạng phân phối. | ❌ | ✅ |

> [!NOTE]
> Khoảng cách giữa SQ tối ưu và cận Shannon cho nguồn Gauss chính là hằng số $\frac{\sqrt 3 \pi}{2} \approx 2.72$ (sẽ chứng minh ở Bài 3 & Bài 6). **Đây chính là con số 2.7 huyền thoại** mà TurboQuant tuyên bố!

### 4.3. Nghịch lý chi phí của VQ

Nếu VQ luôn tốt hơn, vì sao không ai dùng VQ thuần? Vì **chi phí**:
1. **Indexing/Training đắt đỏ**: phải chạy K-means trên toàn bộ dữ liệu (hàng giờ với dataset lớn).
2. **Phụ thuộc dữ liệu (data-dependent)**: codebook học từ một phân phối; khi dữ liệu đổi (distribution shift) thì méo tăng vọt — **không dùng được online**.
3. **Tra cứu chậm**: với KV Cache sinh ra token-by-token, ta không thể chạy K-means lại sau mỗi token.

> 🧠 **Câu hỏi trung tâm của TurboQuant**: *Liệu có cách nào đạt được chất lượng gần như VQ tối ưu, nhưng với chi phí gần như bằng 0 và hoàn toàn data-oblivious (online) như SQ?*
>
> **Câu trả lời**: Có — bằng cách **xoay ngẫu nhiên rồi lượng hóa vô hướng**. Đó là toàn bộ ý tưởng của TurboQuant, sẽ được trình bày từ Bài 2.

---

## 5. Tổng kết Bài 0

* Quantization là cân bằng giữa **Rate** $R$ và **Distortion** $D$.
* Có **hai loại distortion**: MSE (tái tạo vector) và Inner Product (attention/MIPS). TurboQuant lo cả hai.
* Cận Shannon cho nguồn Gauss: $D(R) = \sigma^2 2^{-2R}$ — quy luật **6 dB/bit**.
* **VQ tốt hơn SQ** nhờ space-filling + shape gain, nhưng **đắt và data-dependent**.
* TurboQuant tìm cách "ăn gian": **xoay ngẫu nhiên** để biến bài toán VQ khó thành nhiều bài toán SQ dễ, mà vẫn gần tối ưu.

👉 Bài tiếp theo: **[Bài 1 — Bài toán nén KV Cache & yêu cầu Data-Oblivious](./lesson_1_kv_cache_problem.md)**, đặt TurboQuant vào đúng bối cảnh ứng dụng nóng nhất của nó.
