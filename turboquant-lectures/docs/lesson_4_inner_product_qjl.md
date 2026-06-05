---
sidebar_position: 5
sidebar_label: "Bài 4: Inner Product & QJL Unbiased"
---

# Bài 4: Trụ cột 3 — Inner Product Distortion & QJL

Hai trụ cột đầu (random rotation + MSE quantizer) đã cho ta một bộ nén **tối ưu MSE**. Nhưng như đã cảnh báo ở Bài 0, **tối ưu MSE chưa đủ** cho Attention: thứ Attention cần là **tích vô hướng** $\langle q, k\rangle$, và bộ lượng hóa MSE lại gây **thiên lệch (bias)** trên đại lượng này. Bài này giải thích bias đó từ đâu ra và cách TurboQuant khử nó bằng **Quantized Johnson–Lindenstrauss (QJL)**.

---

## 1. Vì sao MSE-quantizer thiên lệch cho Inner Product?

Gọi $\hat x = Q(x)$ là vector tái tạo, và $e = \hat x - x$ là sai số lượng hóa. Với một query $q$:

$$\langle q, \hat x\rangle = \langle q, x\rangle + \langle q, e\rangle.$$

Ước lượng tốt đòi hỏi sai số $\langle q, e\rangle$ có **kỳ vọng bằng 0** (unbiased). Vấn đề: quantizer **tối ưu MSE** không đảm bảo điều đó. Trực giác:

> [!IMPORTANT]
> Một quantizer MSE tối ưu **co (shrink)** các giá trị về phía trọng tâm ô của chúng để giảm phương sai. Phép co này mang tính **hệ thống** (luôn cùng một hướng), nên $\mathbb E[\hat x] \neq x$ — sai số có thành phần **bias** chứ không chỉ nhiễu ngẫu nhiên. Khi chiếu lên $q$, bias này **không tự triệt tiêu**; tệ hơn, nó **tích lũy** khi cộng qua nhiều token trong attention ($\sum_i a_i \langle q, v_i\rangle$), làm lệch toàn bộ đầu ra.

Nói cách khác: MSE đo $\mathbb E\lVert e\rVert^2$ (độ lớn sai số), nhưng inner product cần $\mathbb E[\langle q,e\rangle] = 0$ (sai số **không lệch**). Hai mục tiêu này khác nhau, và tối ưu cái thứ nhất làm hỏng cái thứ hai.

---

## 2. Ý tưởng: ước lượng KHÔNG thiên lệch quan trọng hơn

Trong ước lượng thống kê, sai số phân rã thành **bias** và **variance**:

$$\mathbb E\big[(\langle q,\hat x\rangle - \langle q,x\rangle)^2\big] = \underbrace{\big(\text{bias}\big)^2}_{\text{systematic}} + \underbrace{\text{variance}}_{\text{random noise}}$$

trong đó **bias** là độ lệch hệ thống (không tự triệt tiêu) còn **variance** là nhiễu ngẫu nhiên (giảm khi trung bình hóa).

Variance giảm khi ta trung bình qua nhiều chiều/nhiều token (luật số lớn), nhưng **bias thì không bao giờ tự biến mất**. Vì vậy với attention — nơi ta cộng rất nhiều inner product — một ước lượng **unbiased dù variance lớn hơn một chút** vẫn tốt hơn nhiều một ước lượng **biased**.

> 🎯 **Mục tiêu của TurboQuant cho inner product**: xây một ước lượng $\widehat{\langle q,x\rangle}$ sao cho $\mathbb E[\widehat{\langle q,x\rangle}] = \langle q,x\rangle$ **chính xác** (unbiased), đổi lại chỉ tốn thêm rất ít bit.

---

## 3. Quantized Johnson–Lindenstrauss (QJL)

**Bổ đề Johnson–Lindenstrauss (JL)** nói rằng: chiếu vector qua một ma trận ngẫu nhiên Gauss $S$ bảo toàn (kỳ vọng) tích vô hướng:

$$\mathbb E\big[\langle Sq, Sx\rangle\big] = \langle q, x\rangle.$$

**QJL** (Quantized JL, một kỹ thuật cũng từ nhóm Zandieh và cộng sự, dùng trong KVQuant/QJL trước đó) đẩy ý tưởng này tới cực hạn: chỉ giữ lại **dấu (sign)** của các phép chiếu — tức **1 bit mỗi chiều chiếu**:

$$\text{QJL}(x) = \operatorname{sign}(S x) \in \{-1, +1\}^{m}.$$

Điều kỳ diệu: với chuẩn hóa thích hợp, ước lượng dựa trên dấu này **vẫn không thiên lệch** cho inner product (liên hệ với công thức góc $\langle q,x\rangle \propto \cos\theta$, và xác suất hai dấu trùng nhau là $1 - \theta/\pi$ — chính là **SimHash / sign random projection**). QJL cho ta một **ước lượng unbiased với chỉ 1 bit/chiều**.

---

## 4. Sơ đồ hai pha của TurboQuant cho Inner Product

TurboQuant **kết hợp** hai thế mạnh: độ chính xác tái tạo của MSE-quantizer và tính unbiased của QJL, theo một sơ đồ **hai pha trên phần dư (residual)**:

```text
            ┌────────────────────────── PHA 1 ──────────────────────────┐
  x  ──►  [Random Rotation]  ──►  [MSE Scalar Quantizer Q]  ──►  x̂ = Q(x)
                                                                  │
                                          residual  r = x − x̂  ◄──┘
            ┌────────────────────────── PHA 2 ──────────────────────────┐
  r  ──►  [QJL: 1-bit sign random projection]  ──►  mã dấu (khử bias)
```

* **Pha 1 (MSE quantizer)**: nén phần lớn năng lượng vector với méo nhỏ (Bài 3). Phần này gánh độ chính xác MSE nhưng còn **bias** ở inner product.
* **Pha 2 (1-bit QJL trên residual $r = x - \hat x$)**: lượng hóa **phần dư** bằng QJL. Vì QJL là **unbiased**, nó **bù trừ chính xác thành phần bias** mà pha 1 để lại.

Kết quả tổng hợp:

$$\widehat{\langle q,x\rangle} = \underbrace{\langle q, \hat x\rangle}_{\text{pha 1}} + \underbrace{\widehat{\langle q, r\rangle}_{\text{QJL}}}_{\text{pha 2, unbiased}}, \qquad \mathbb E\big[\widehat{\langle q,x\rangle}\big] = \langle q,x\rangle.$$

> [!IMPORTANT]
> Đây là đóng góp lý thuyết tinh tế nhất của paper: *"applying an MSE quantizer followed by a 1-bit Quantized JL transform on the residual, resulting in an unbiased inner product quantizer."* Chỉ tốn thêm **~1 bit/kênh**, TurboQuant biến một ước lượng biased thành **unbiased** — đúng thứ Attention cần.

---

## 5. Hai chế độ của TurboQuant

Tóm lại, TurboQuant có hai biến thể, dùng tùy mục tiêu:

| Chế độ | Thành phần | Tối ưu cho | Dùng khi |
| :--- | :--- | :--- | :--- |
| **MSE mode** | Rotation + MSE quantizer | $\mathbb E\lVert x-\hat x\rVert^2$ | Tái tạo vector chính xác, nén Value cache |
| **Inner-product mode** | Rotation + MSE quantizer + **QJL residual** | $\mathbb E[\langle q,\hat x\rangle]$ unbiased | Attention scores, MIPS, nén Key cache |

> [!TIP]
> Trong ngữ cảnh vLLM (Bài 5): **Key cache** dùng để tính điểm attention $\langle q, k\rangle$ → hưởng lợi từ **inner-product mode (có QJL)**; **Value cache** dùng để tính tổ hợp đầu ra → có thể dùng **MSE mode** đơn giản hơn. Việc tách riêng này giúp tiết kiệm bit ở chỗ không cần unbiased.

---

## 6. Tổng kết Bài 4

* Quantizer **tối ưu MSE gây bias** cho inner product vì nó co giá trị một cách hệ thống; bias **tích lũy** qua attention, không tự triệt tiêu.
* Với inner product, **unbiased quan trọng hơn variance nhỏ** — vì bias không giảm khi trung bình.
* **QJL** = sign random projection cho **ước lượng unbiased chỉ với 1 bit/chiều** (họ hàng SimHash).
* TurboQuant dùng **sơ đồ hai pha**: MSE quantizer + **1-bit QJL trên residual** → unbiased inner product, tốn thêm ~1 bit.
* Phân biệt **MSE mode** (Value) và **inner-product mode** (Key) khi triển khai.

👉 Bài tiếp theo: **[Bài 5 — Tích hợp TurboQuant vào vLLM](./lesson_5_vllm_integration.md)** — đưa toàn bộ lý thuyết vào thực tế serving.
