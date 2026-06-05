---
sidebar_position: 3
sidebar_label: "Bài 2: Random Rotation & Phân phối Beta"
---

# Bài 2: Trụ cột 1 — Random Rotation & Phân phối Beta

Đây là trái tim của TurboQuant. Toàn bộ "phép màu" nằm ở một quan sát toán học tuyệt đẹp: **nếu bạn xoay ngẫu nhiên một vector ở chiều cao, mọi tọa độ của nó sẽ tuân theo cùng một phân phối đã biết, và gần như độc lập với nhau.** Quan sát này biến bài toán vector quantization khó thành $d$ bài toán scalar quantization dễ.

---

## 1. Phép xoay ngẫu nhiên (Random Rotation)

Cho vector $x \in \mathbb{R}^d$. Ta chọn một **ma trận trực giao ngẫu nhiên** $R \in \mathbb{R}^{d\times d}$ (tức $R^\top R = I$) và biến đổi:

$$\tilde{x} = R\,x.$$

Vì $R$ trực giao, phép xoay **bảo toàn mọi cấu trúc hình học** quan trọng:

$$\lVert \tilde{x} \rVert_2 = \lVert x \rVert_2, \qquad \langle Rx, Ry\rangle = \langle x, y\rangle.$$

> [!IMPORTANT]
> Đây là tính chất vàng: phép xoay **không làm mất thông tin** (có thể đảo ngược bằng $R^\top$) và **bảo toàn cả chuẩn lẫn tích vô hướng** — chính là hai đại lượng ta cần giữ chính xác cho KV Cache (MSE và attention scores). Ta được "miễn phí" đổi hệ tọa độ sang một hệ thuận lợi hơn cho việc lượng hóa.

Phía giải mã chỉ cần áp $R^\top$ để xoay ngược về. Vì $R$ là **ngẫu nhiên nhưng được chia sẻ** (encoder và decoder dùng cùng seed), ta **không cần lưu** ma trận — chỉ cần lưu seed. Đây là yếu tố then chốt khiến nó **data-oblivious**.

---

## 2. Phân phối của tọa độ sau khi xoay

Không mất tính tổng quát, xét $x$ là **vector đơn vị** (ta tách riêng độ lớn $\lVert x\rVert$ và lượng hóa nó bằng vài bit — xem Bài 3). Sau khi xoay ngẫu nhiên, $\tilde{x} = Rx$ là **một điểm phân bố đều trên mặt cầu đơn vị** $\mathbb{S}^{d-1}$.

Câu hỏi: mỗi tọa độ $\tilde{x}_i$ có phân phối gì? Đây là kết quả cổ điển của hình học chiều cao:

$$\boxed{\;\tilde{x}_i^2 \sim \text{Beta}\!\left(\tfrac{1}{2},\, \tfrac{d-1}{2}\right)\;}$$

Tương đương, bản thân $\tilde{x}_i$ có mật độ tỉ lệ với $(1 - t^2)^{(d-3)/2}$ trên $[-1, 1]$. Hai hệ quả quan trọng:

* **Kỳ vọng & phương sai**: $\mathbb{E}[\tilde{x}_i] = 0$ và $\mathbb{E}[\tilde{x}_i^2] = \dfrac{1}{d}$. Năng lượng được **chia đều** cho $d$ tọa độ.
* **Xấp xỉ Gauss ở chiều cao**: khi $d$ lớn, theo định lý giới hạn,
  $$\sqrt{d}\,\tilde{x}_i \;\xrightarrow{\;d\to\infty\;}\; \mathcal{N}(0, 1).$$
  Nghĩa là mỗi tọa độ (sau khi nhân $\sqrt d$) **xấp xỉ chuẩn tắc** $\mathcal{N}(0,1)$.

```
   Phân phối tọa độ TRƯỚC khi xoay          SAU khi xoay (mọi tọa độ giống nhau)
   (có outlier, mỗi kênh một kiểu)          ~ Beta ≈ Gauss, không outlier

   |        █                                         ▁▃▅█▅▃▁
   |  ▁▁    █    ▁▁                                 ▁▃█████████▃▁
   |▁███▁▁▁▁█▁▁▁████▁          ──xoay──►          ▁████████████████▁
   +─────────────────                            ───────────────────
    kênh đặc biệt lớn = outlier                   đối xứng, tập trung quanh 0
```

> [!TIP]
> Đây chính là cơ chế **"diệt outlier"** đã hứa ở Bài 1: một kênh outlier khổng lồ trong $x$ sau khi nhân với một hàng ngẫu nhiên của $R$ sẽ bị **tán xạ thành tổ hợp tuyến tính ngẫu nhiên** của tất cả các kênh, làm phẳng phân phối. Không còn tọa độ nào "đặc biệt" — mọi tọa độ thống kê **như nhau**.

---

## 3. Tính gần-độc-lập ở chiều cao (Near-Independence)

Quan sát thứ hai, cũng then chốt không kém: ở chiều cao, các tọa độ $\tilde{x}_i$ và $\tilde{x}_j$ ($i\neq j$) **gần như độc lập thống kê**. Ràng buộc duy nhất nối chúng là $\sum_i \tilde{x}_i^2 = 1$ (nằm trên mặt cầu), nhưng ràng buộc này "loãng dần" khi $d$ tăng — mỗi tọa độ chỉ đóng góp $O(1/d)$.

**Hệ quả "chia để trị"**: Vì các tọa độ (a) có **cùng phân phối** đã biết và (b) **gần như độc lập**, bài toán lượng hóa vector $d$ chiều **phân rã** thành $d$ bài toán lượng hóa vô hướng **giống hệt nhau và độc lập**:

$$\min_{Q} \; \mathbb{E}\lVert \tilde{x} - Q(\tilde{x})\rVert^2 \;\approx\; \sum_{i=1}^{d} \min_{q} \mathbb{E}\big[(\tilde{x}_i - q(\tilde{x}_i))^2\big].$$

Ta chỉ cần thiết kế **một** bộ lượng hóa vô hướng tối ưu duy nhất $q^\star$ cho phân phối Beta/Gauss, rồi áp nó cho **mọi** tọa độ. Đó chính là nội dung Bài 3.

> [!NOTE]
> Đây là lý do TurboQuant **near-optimal**: lượng hóa vô hướng độc lập thường mất "shape gain" (Bài 0), nhưng vì phép xoay đã **khử mọi tương quan** và làm các tọa độ độc lập-đồng-phân-phối (i.i.d.), nên phần lớn shape gain biến mất một cách tự nhiên — không còn gì để khai thác. Cái duy nhất còn thiếu là **space-filling gain**, và đó chính xác là hằng số gap $\approx 2.72$ ở Bài 6.

---

## 4. Hiện thực hiệu quả: Randomized Hadamard Transform (RHT)

Một ma trận trực giao ngẫu nhiên $R$ tổng quát tốn $O(d^2)$ phép tính để nhân — quá đắt cho mỗi vector KV ở mỗi token. TurboQuant (và các phương pháp họ Hadamard) dùng một **xấp xỉ rẻ hơn nhiều**: **Randomized Hadamard Transform**.

$$R = \frac{1}{\sqrt{d}}\, H D$$

trong đó:
* $H$ là **ma trận Hadamard** $d\times d$ (các phần tử $\pm 1$), nhân được trong $O(d\log d)$ bằng thuật toán **Fast Walsh–Hadamard Transform (FWHT)** — không cần lưu ma trận.
* $D = \text{diag}(s_1, \dots, s_d)$ với $s_i \in \{+1, -1\}$ là các **dấu ngẫu nhiên** (random sign flip). Đây là phần "ngẫu nhiên hóa" để mỗi vector được xoay khác nhau và phá vỡ cấu trúc cố định của $H$.

| Phương pháp | Chi phí/vector | Lưu trữ ma trận |
| :--- | :--- | :--- |
| Ma trận trực giao đầy đủ | $O(d^2)$ | $O(d^2)$ |
| **Randomized Hadamard (RHT)** | $O(d\log d)$ | $O(d)$ (chỉ các dấu $s_i$) |

> [!IMPORTANT]
> RHT cho ta chất lượng "xoay ngẫu nhiên" gần như hoàn hảo (tọa độ vẫn xấp xỉ Beta/Gauss và gần độc lập) với chi phí **gần tuyến tính** $O(d\log d)$. Đây là điều khiến TurboQuant đủ rẻ để chạy **online trong vòng lặp decode** của vLLM. Ta sẽ hiện thực FWHT thật ở Bài 8.

---

## 5. Tóm tắt đường ống (Encoder)

```text
x  ──►  [Random Rotation R = HD/√d]  ──►  x̃ (mỗi tọa độ ~ Beta ≈ Gauss, i.i.d.)
                                            │
                                            ▼
                            [Scalar Quantizer q* cho từng tọa độ]   ← Bài 3
                                            │
                                            ▼
                                   mã b bit/tọa độ
```

Giải mã đảo ngược: dequantize từng tọa độ → nhân $R^\top = D H^\top/\sqrt d$ để xoay về hệ gốc.

---

## 6. Tổng kết Bài 2

* **Random rotation** $\tilde x = Rx$ bảo toàn chuẩn và tích vô hướng, có thể đảo ngược, **không cần lưu ma trận** (chỉ cần seed) → data-oblivious.
* Sau khi xoay, mọi tọa độ tuân theo **cùng phân phối Beta** ($\tilde x_i^2 \sim \text{Beta}(\tfrac12, \tfrac{d-1}2)$), xấp xỉ **Gauss** $\mathcal N(0, 1/d)$ ở chiều cao, và **gần như độc lập**.
* Điều này **diệt outlier** và cho phép **phân rã** bài toán VQ thành $d$ bài toán SQ giống hệt nhau.
* Hiện thực rẻ bằng **Randomized Hadamard Transform** $O(d\log d)$.

👉 Bài tiếp theo: **[Bài 3 — MSE Scalar Quantizer tối ưu](./lesson_3_mse_scalar_quantizer.md)**, nơi ta thiết kế bộ lượng hóa vô hướng $q^\star$ và tính chính xác méo của nó.
