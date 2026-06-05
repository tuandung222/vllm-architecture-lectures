---
sidebar_position: 11.2
sidebar_label: "Bài 7.2: Toán học & Cache trong Speculative Decoding"
---

# Bài 7.2: Giải mã Suy đoán (Speculative Decoding) - Bản chất Toán học & Phân bổ Cache

Trong pha Decode của LLM, nút thắt hiệu năng nằm ở băng thông bộ nhớ (Memory-bound) do GPU phải nạp toàn bộ trọng số mô hình khổng lồ từ HBM lên SRAM chỉ để sinh ra đúng **1 token** ở mỗi bước.

**Giải mã suy đoán (Speculative Decoding)** là kỹ thuật tối ưu hóa đột phá giúp phá vỡ giới hạn này. Bằng cách sử dụng phối hợp một mô hình nhỏ chạy nhanh (Draft Model) và một mô hình lớn chạy chậm nhưng chính xác (Target Model), chúng ta có thể sinh ra **nhiều token trong một bước forward duy nhất của mô hình lớn**.

Bài học này sẽ mổ xẻ bản chất toán học của thuật toán **Rejection Sampling** và cách vLLM quản lý, thu hồi KV Cache động trong quá trình suy đoán.

---

## 1. Nguyên lý Hoạt động: Vòng lặp Suy đoán - Xác thực (Draft-Verify Loop)

Ý tưởng cốt lõi của Speculative Decoding dựa trên một thực tế: **Xác thực song song nhiều token dễ hơn và nhanh hơn rất nhiều so với việc sinh từng token tuần tự**.

Quy trình hoạt động diễn ra qua 3 bước:
1.  **Suy đoán (Drafting)**: Mô hình nhỏ (Draft Model, ví dụ Llama-3-Draft-1B) chạy tuần tự $K$ bước sinh token (ví dụ $K = 4$), tạo ra một chuỗi nháp gồm $K$ tokens: $x_1, x_2, \dots, x_K$. Bước này cực nhanh vì mô hình nhỏ có dung lượng weights bé, ít nghẽn băng thông bộ nhớ.
2.  **Xác thực (Verification)**: Chuỗi $K$ tokens nháp được đóng gói và đưa vào mô hình lớn (Target Model, ví dụ Llama-3-8B) để chạy **đúng 1 bước forward duy nhất**. Do xử lý đồng thời $K$ tokens, đây là phép toán song song tính toán (GEMM - Compute-bound) giúp tận dụng tối đa sức mạnh của GPU.
3.  **Đồng bộ & Cập nhật**: Mô hình lớn sẽ tính toán phân phối xác suất thực tế của các token này và quyết định chấp nhận (Accept) bao nhiêu token nháp, bác bỏ (Reject) các token sai và sinh ra 1 token mới chuẩn xác.

---

## 2. Bản chất Toán học: Thuật toán Rejection Sampling

Để đảm bảo kết quả đầu ra của Speculative Decoding có **phân phối xác suất hoàn toàn trùng khớp với việc chạy trực tiếp mô hình lớn** (không làm suy giảm chất lượng văn bản), hệ thống sử dụng thuật toán **Rejection Sampling (Lấy mẫu bác bỏ)**.

Gọi:
* $q(x)$: Xác suất sinh ra token $x$ của mô hình nhỏ (Draft Model).
* $p(x)$: Xác suất sinh ra token $x$ của mô hình lớn (Target Model).
* Chuỗi nháp đề xuất: $x_1, x_2, \dots, x_K$.

### 2.1. Quy trình Kiểm thử Bác bỏ (Verification Steps)
Với mỗi token nháp $x_i$ (từ $i = 1$ đến $K$), chúng ta tính toán tỷ lệ xác thực:
$$\alpha = \min\left(1, \frac{p(x_i)}{q(x_i)}\right)$$

Hệ thống sẽ lấy mẫu một số ngẫu nhiên $r \sim U(0, 1)$:
*   **Chấp nhận (Accept)**: Nếu $r \le \alpha$, token nháp $x_i$ được giữ lại. Chúng ta tiếp tục kiểm tra token tiếp theo $x_{i+1}$.
*   **Bác bỏ (Reject)**: Nếu $r > \alpha$, token nháp $x_i$ bị từ chối. Ngay lập tức, chúng ta **dừng việc xác thực**, loại bỏ toàn bộ các token nháp từ vị trí đó về sau ($x_i, x_{i+1}, \dots, x_K$).

### 2.2. Lấy mẫu lại (Resampling) khi bị Bác bỏ
Khi token nháp $x_i$ bị bác bỏ tại bước $j$, mô hình lớn sẽ sinh ra một token thay thế $x'_j$ bằng cách lấy mẫu từ phân phối xác suất đã được hiệu chỉnh:
$$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{y} \max(0, p(y) - q(y))}$$

Phân phối này đảm bảo bù đắp lại phần xác suất mà mô hình nhỏ đã bỏ sót hoặc ước lượng sai.

* **Trường hợp lý tưởng**: Nếu cả $K$ tokens đều được chấp nhận, mô hình lớn sẽ tự động sinh thêm miễn phí token thứ $K+1$ từ phân phối $p(x_{K+1})$ ở cuối bước forward. Tổng cộng ta có được $K+1$ tokens.
* **Trường hợp tệ nhất**: Token đầu tiên $x_1$ bị bác bỏ ngay lập tức, ta chỉ thu được $1$ token thay thế $x'_1$ từ phân phối hiệu chỉnh.

---

## 3. Quản lý KV Cache và Thu hồi Block (Cache Rollback)

Thách thức lớn nhất khi hiện thực Speculative Decoding trong các engine phục vụ như vLLM là **quản lý và đồng bộ hóa KV Cache giữa hai mô hình**.

Khi mô hình nháp thực hiện $K$ bước sinh thử, nó phải ghi các khóa Key-Value tạm thời của $K$ token này vào hệ thống KV Cache. 

```
1. Pha Draft sinh K=3 tokens:
[Block 0 (Đã xác thực)] ➔ [Block 1 (Nháp 1)] ➔ [Block 2 (Nháp 2)] ➔ [Block 3 (Nháp 3)]

2. Pha Verify bác bỏ tại Token 2:
[Block 0 (Đã xác thực)] ➔ [Block 1 (Chấp nhận 1)] ➔ [HỦY Block 2 và 3!]
```

### Cơ chế Rollback (Thu hồi bộ nhớ) của vLLM:
vLLM giải quyết bài toán này thông qua lớp quản lý khối ảo **Block Manager**:
1.  **Cấp phát tạm thời (Speculative Allocation)**: Khi Draft model chạy, Block Manager cấp phát các khối vật lý tạm thời cho các token nháp. Các block này được đánh dấu ở trạng thái "chờ xác thực".
2.  **Xác nhận (Commit)**: Khi Target model xác thực và chấp nhận $j$ tokens đầu tiên, Block Manager chuyển trạng thái của $j$ khối tạm thời này thành "đã xác thực" (chính thức ghi nhận vào KV Cache lịch sử).
3.  **Thu hồi (Rollback)**: Đối với các khối chứa token bị bác bỏ từ vị trí $j+1$ đến $K$, Block Manager lập tức thu hồi các con trỏ địa chỉ, trả các khối vật lý này về **VRAM Block Pool** tự do để cấp phát cho các request khác.
4.  Quá trình rollback này được thực hiện tức thì ở mức siêu dữ liệu (Metadata) thông qua bảng ánh xạ **Block Table**, hoàn toàn không tốn chi phí copy hay dọn dẹp vật lý trên GPU VRAM.

---

## 💡 Tổng kết bài học

* **Speculative Decoding** tận dụng mô hình nhỏ chạy nhanh để suy đoán và mô hình lớn chạy song song tính toán (GEMM) để xác thực nhiều token cùng lúc.
* Thuật toán **Rejection Sampling** đảm bảo tính chính xác tuyệt đối của phân phối xác suất đầu ra so với mô hình lớn chạy đơn lẻ.
* Số lượng token sinh ra ở mỗi bước biến thiên từ $1$ đến $K+1$ tùy thuộc vào độ chính xác của mô hình nháp.
* Sự thành bại của kỹ thuật này phụ thuộc vào khả năng quản lý và **Rollback KV Cache** cực nhanh của bộ quản lý khối ảo (Block Manager) trong vLLM mà không gây ra overhead về bộ nhớ.
