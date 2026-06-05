---
sidebar_position: 5.5
sidebar_label: "Bài 3.2: Chunked Prefill & Mixed Batching"
---

# Bài 3.2: Kỹ thuật Chunked Prefill & Lập lịch Hỗn hợp (Mixed-Batch Scheduling)

Trong [Bài 3](./lesson_3_continuous_batching.md), chúng ta đã tìm hiểu về **Continuous Batching** và cách nó cách mạng hóa hiệu suất phục vụ LLM bằng cách lập lịch ở mức độ Iteration. Tuy nhiên, khi hệ thống phải đối mặt với các Prompt có độ dài cực lớn (như ngữ cảnh RAG lên tới 8K, 16K hoặc 32K tokens), Continuous Batching tiêu chuẩn bắt đầu bộc lộ một điểm yếu nghiêm trọng: **Hiện tượng giật cục độ trễ (Latency Spikes)**.

Bài học này sẽ phân tích chi tiết nguyên nhân vật lý của hiện tượng này và giải pháp khắc phục triệt để của vLLM: **Chunked Prefill** kết hợp với **Mixed-Batch Scheduling**.

---

## 1. Bản chất Vật lý của Latency Spikes trong LLM Serving

Để hiểu tại sao prompt dài lại tàn phá độ trễ của hệ thống, chúng ta phải quay lại sự khác biệt về tính toán giữa pha **Prefill** và **Decode**:

*   **Pha Decode**: Xử lý 1 token đầu vào ở mỗi bước. Đây là phép toán nhân ma trận-vector (**GEMV**), bị nghẽn bởi băng thông bộ nhớ (**Memory-bound**). Thời gian chạy 1 bước Decode rất nhanh và ổn định (ví dụ: $10 - 20\text{ ms}$).
*   **Pha Prefill**: Xử lý toàn bộ Prompt đầu vào song song. Đây là phép toán nhân ma trận-ma trận (**GEMM**), tận dụng tối đa các Tensor Core của GPU nên bị nghẽn bởi năng lực tính toán (**Compute-bound**).

### Mối quan hệ giữa Chiều dài Prompt và Thời gian Tính toán:
Độ phức tạp tính toán của cơ chế Self-Attention tỷ lệ thuận với **bình phương** chiều dài chuỗi đầu vào ($O(L^2)$). Khi chiều dài Prompt $L$ tăng lên, thời gian forward của pha Prefill tăng theo hàm mũ:

$$\text{Time}_{\text{prefill}}(L) \propto \alpha \cdot L + \beta \cdot L^2$$

Trong đó:
*   $\alpha \cdot L$: Chi phí tính toán các lớp tuyến tính (Linear Projections Q, K, V, MLP).
*   $\beta \cdot L^2$: Chi phí tính toán ma trận tương quan Attention ($Q K^T$).

| Chiều dài Prompt ($L$) | Thời gian GPU xử lý Prefill (Ước lượng) | Trạng thái nghẽn phần cứng |
| :--- | :--- | :--- |
| **512 tokens** | $\approx 10\text{ ms}$ | Trọng số mô hình chiếm chủ đạo |
| **2048 tokens** | $\approx 40\text{ ms}$ | Bắt đầu chuyển sang Compute-bound |
| **8192 tokens** | $\approx 250 - 400\text{ ms}$ | Hoàn toàn Compute-bound |
| **32768 tokens** | $\approx 1500 - 3000\text{ ms}$ | Nghẽn nghiêm trọng, khóa cứng GPU |

### Kịch bản Latency Spike:
Hãy tưởng tượng hệ thống đang chạy Continuous Batching phục vụ 4 request đang ở pha Decode (độ trễ mỗi bước chỉ $\approx 15\text{ ms}$). 
Đột nhiên, một request mới với prompt dài **8000 tokens** xuất hiện. 
*   Nếu không có Chunked Prefill, bộ lập lịch sẽ nạp toàn bộ prompt 8000 tokens này vào GPU để chạy prefill ở iteration tiếp theo.
*   GPU bị "khóa cứng" trong **300ms** để hoàn thành phép toán khổng lồ này.
*   **Hậu quả**: 4 request decode đang chạy mượt mà bị chặn lại hoàn toàn (block) trong suốt 300ms này. Người dùng đang nhận stream text sẽ cảm nhận rõ rệt việc văn bản dừng lại đột ngột trong gần 1/3 giây. Chỉ số **Inter-Token Latency (ITL)** của hệ thống bị nhảy vọt (spike).

---

## 2. Giải pháp: Chunked Prefill & Lập Lịch Hỗn Hợp (Mixed-Batching)

Ý tưởng cốt lõi của **Chunked Prefill** là: **Thay vì nạp toàn bộ prompt dài một lần, chúng ta chia nhỏ prompt đó thành các phần (chunk) có kích thước cố định $C$ (ví dụ: $C = 512$ hoặc $1024$ tokens) và xử lý chúng qua nhiều bước (iterations) liên tiếp.**

### Sơ đồ so sánh luồng thực thi:

#### 1. Không dùng Chunked Prefill (Standard Continuous Batching)
```
Iteration 1 (Prefill Req A): [=================== Req A: 8000 tokens ===================] ➔ Mất 300ms! (Req B, C bị block)
Iteration 2 (Decode All)   : [ Decode A ] + [ Decode B ] + [ Decode C ]                   ➔ Mất 20ms
```

#### 2. Có dùng Chunked Prefill (C = 512)
```
Iteration 1: [ Chunk 1 of A (512 t) ] + [ Decode B ] + [ Decode C ] ➔ Mất 25ms
Iteration 2: [ Chunk 2 of A (512 t) ] + [ Decode B ] + [ Decode C ] ➔ Mất 25ms
...
Iteration 16: [ Chunk 16 of A (512 t) ] + [ Decode B ] + [ Decode C ] ➔ Mất 25ms (Hoàn thành Prefill Req A)
Iteration 17: [ Decode A ] + [ Decode B ] + [ Decode C ]             ➔ Mất 20ms
```
Nhờ cơ chế này, ở mỗi iteration, GPU chỉ phải xử lý một lượng token giới hạn. Độ trễ của mỗi bước được khống chế ở mức cực thấp và ổn định ($\approx 25\text{ ms}$), loại bỏ hoàn toàn hiện tượng đóng băng hệ thống.

---

## 3. Bản chất Tính toán KV Cache trong Chunked Prefill

Khi chia nhỏ prompt thành nhiều chunk, làm thế nào để mô hình tính toán cơ chế Attention chính xác cho các chunk phía sau? 

Giả sử chúng ta đang ở **Chunk thứ $i$** (tương ứng với các token từ vị trí $i \cdot C$ đến $(i+1) \cdot C - 1$):
1.  **Q (Query)**: Chỉ được tạo ra từ các token của **Chunk thứ $i$ hiện tại** (Kích thước $C \times d$).
2.  **K, V (Key, Value)**: Được tạo ra từ toàn bộ prompt tính từ đầu đến vị trí hiện tại. Nghĩa là $K$ và $V$ bao gồm cả **KV Cache đã tính toán và lưu trong VRAM từ các chunk trước đó ($1 \dots i-1$)** nối tiếp với KV của Chunk hiện tại.
3.  **Attention Operator**: Phép toán Attention lúc này thực hiện nhân ma trận Query của chunk hiện tại với toàn bộ Key, Value lịch sử:
    $$\text{Attention}(\text{Chunk}_i) = \text{Softmax}\left(\frac{Q_i \cdot [K_{1\dots i-1} ; K_i]^T}{\sqrt{d}}\right) \cdot [V_{1\dots i-1} ; V_i]$$

### Cấp phát Block vật lý động (Incremental Block Allocation):
Trình quản lý khối ảo `BlockManager` phối hợp chặt chẽ trong quá trình này:
*   Thay vì cấp phát toàn bộ các block vật lý cho cả prompt 8000 tokens ngay lập tức, `BlockManager` chỉ cấp phát block cho chunk đầu tiên.
*   Ở mỗi bước xử lý chunk tiếp theo, `BlockManager` cấp phát thêm các block vật lý mới một cách tuần tự (Incremental) để chứa các Key-Value vừa được sinh ra từ chunk mới.
*   Điều này giúp tối ưu hóa dung lượng VRAM: bộ nhớ được tiêu thụ dần dần, tránh việc giữ chỗ (reserve) lãng phí một lượng lớn block vật lý trong nhiều chu kỳ khi chúng chưa thực sự được ghi dữ liệu.

---

## 4. Hiện thực Lập lịch trong Mã nguồn vLLM

Trong codebase **vLLM v1**, logic Chunked Prefill được tích hợp trực tiếp vào hàm lập lịch cốt lõi trong file [scheduler.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/core/sched/scheduler.py).

Khi duyệt qua các request đang chờ trong hàng đợi `Waiting Queue`, bộ lập lịch sẽ tính toán số lượng token mới cần nạp dựa trên ngân sách token (`token_budget` - được giới hạn bởi cấu hình phần cứng và tham số tối đa):

```python
# Trích đoạn đơn giản hóa logic lập lịch trong vllm/v1/core/sched/scheduler.py
num_new_tokens = request.num_tokens - num_computed_tokens
threshold = self.scheduler_config.long_prefill_token_threshold

if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold

# Nếu Chunked Prefill BỊ TẮT, và số lượng token cần nạp vượt quá budget còn lại
if (
    not self.scheduler_config.enable_chunked_prefill
    and num_new_tokens > token_budget
):
    # Chúng ta bắt buộc phải dừng lập lịch ở đây (không thể chèn một phần)
    break

# Nếu Chunked Prefill ĐƯỢC BẬT, chúng ta cắt nhỏ request để vừa khít với budget
num_new_tokens = min(num_new_tokens, token_budget)
```

### Cách cấu hình trong vLLM:
Người dùng có thể kích hoạt cơ chế này khi khởi chạy vLLM thông qua tham số dòng lệnh:
```bash
vllm serve facebook/opt-13b --enable-chunked-prefill --max-num-batched-tokens 2048
```
*   `--enable-chunked-prefill`: Bật tính năng chia nhỏ prefill.
*   `--max-num-batched-tokens`: Xác định giới hạn tối đa số lượng token ($C$) được xử lý trong một iteration (bao gồm cả chunk prefill và các token decode).

---

## 💡 Tổng kết bài học

*   **Latency Spikes** xảy ra do các prompt prefill dài có độ phức tạp tính toán lớn ($O(L^2)$) làm nghẽn GPU, trì hoãn các request decode khác.
*   **Chunked Prefill** giải quyết bài toán bằng cách chia prompt dài thành các chunk kích thước nhỏ ($C$), xử lý dần qua nhiều iteration.
*   **Mixed-Batching** cho phép chạy song song chunk prefill và các token decode trong cùng một bước forward, tối đa hóa hiệu suất phần cứng của GPU.
*   Cơ chế này cải thiện đáng kể chỉ số **Inter-Token Latency (ITL)** và độ trễ phân vị cao (P99 latency), giúp hệ thống serving vận hành mượt mà dưới tải lớn.
