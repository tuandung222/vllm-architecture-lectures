---
sidebar_position: 8.97
sidebar_label: "Bài 6.7: Context Parallelism & Ring Attention"
---

# Bài 6.7: Context Parallelism & Ring Attention - Giải pháp xử lý ngữ cảnh siêu dài

Sự bùng nổ của các ứng dụng RAG (Retrieval-Augmented Generation) và phân tích tài liệu dài đòi hỏi các LLM phục vụ các chuỗi văn bản khổng lồ (từ 100K đến hơn 1M tokens). Khi ngữ cảnh đạt tới quy mô này, KV Cache của một request đơn lẻ có thể chiếm tới hàng chục GB VRAM, vượt quá giới hạn vật lý của một GPU đơn.

Bài học này sẽ giải thích tại sao Tensor Parallelism (TP) thông thường thất bại trước bài toán ngữ cảnh siêu dài, mổ xẻ cơ chế song song ngữ cảnh **Context Parallelism (CP)** thông qua mã nguồn vLLM và giải thuật xoay vòng dữ liệu **Ring Attention**.

---

## 1. Tại sao Tensor Parallelism (TP) thất bại trước ngữ cảnh siêu dài?

Trong các mô hình Transformer, dung lượng KV Cache của một request tỷ lệ thuận tuyến tính với độ dài ngữ cảnh ($O(L)$). Khi chạy song song hóa bằng Tensor Parallelism (TP), chúng ta phân chia các đầu Attention (Attention Heads) đều cho các GPU.

Mặc dù số lượng head trên mỗi GPU giảm đi theo tỷ lệ $\frac{H}{\text{TP}}$, nhưng **chiều dài của chuỗi ngữ cảnh $L$ vẫn giữ nguyên** trên tất cả các GPU. Với chuỗi 1 triệu tokens:
*   Mỗi GPU vẫn phải lưu trữ toàn bộ lịch sử ngữ cảnh $L$ để thực hiện phép toán Attention của các heads do nó quản lý.
*   Dung lượng KV Cache khổng lồ này vẫn sẽ nhanh chóng gây tràn bộ nhớ VRAM (Out-Of-Memory) trên từng GPU đơn lẻ, bất kể bạn có tăng cấu hình TP lên bao nhiêu đi chăng nữa.

Để giải quyết triệt để nút thắt này, chúng ta bắt buộc phải phân chia chính **chiều dài chuỗi ngữ cảnh (Sequence Dimension)** của KV Cache trên các GPU vật lý khác nhau. Đây chính là nguyên lý của **Context Parallelism (CP)**.

```
* Tensor Parallelism (TP): Cắt theo chiều ngang (Attention Heads / Hidden Dimension)
* Context Parallelism (CP): Cắt theo chiều dọc (Sequence Length Dimension)
```

---

## 2. DCP (Distributed Context Parallelism) trong vLLM

vLLM v1 hiện thực hóa Context Parallelism dưới tên gọi **DCP** thông qua tệp mã nguồn [cp_utils.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/worker/gpu/cp_utils.py). 

Trọng tâm của DCP là chia nhỏ độ dài chuỗi toàn cục (`seq_lens`) thành các độ dài chuỗi cục bộ (`dcp_local_seq_lens`) cho từng rank GPU vật lý (`dcp_rank`) theo cơ chế chia vòng tròn xen kẽ (interleaved round-robin).

Hãy cùng mổ xẻ Triton kernel thực hiện phân bổ này:

```python
# Trích từ cp_utils.py: _dcp_local_seq_lens_kernel
@triton.jit
def _dcp_local_seq_lens_kernel(
    out_ptr,
    seq_lens_ptr,
    dcp_size,      # Số lượng GPU tham gia nhóm CP
    dcp_rank,      # Thứ hạng GPU hiện tại (0 đến dcp_size-1)
    cp_interleave, # Kích thước khối sequence xen kẽ
    num_reqs,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    seq_lens = tl.load(seq_lens_ptr + block, mask=block < num_reqs)

    # Chia chuỗi ngữ cảnh thành các vòng tròn xen kẽ
    rounds = seq_lens // (dcp_size * cp_interleave)
    remainder = seq_lens % (dcp_size * cp_interleave)

    remainder = tl.maximum(remainder - dcp_rank * cp_interleave, 0)
    remainder = tl.minimum(remainder, cp_interleave)
    
    # Tính toán độ dài chuỗi cục bộ trên GPU rank này
    local_seq_lens = rounds * cp_interleave + remainder
    
    # Ghi nhận kết quả
    local_seq_lens = tl.where(block < num_reqs, local_seq_lens, 0)
    tl.store(out_ptr + block, local_seq_lens, mask=block < max_num_reqs)
```

Cơ chế phân bổ xen kẽ `cp_interleave` này đảm bảo lượng KV Cache và lượng tính toán của các request trong batch luôn được trải đều trên các GPU, hạn chế tối đa hiện tượng lệch tải (load imbalance) khi chiều dài sequence của các request biến động.

---

## 3. Giải thuật Ring Attention và cơ chế truyền tin xoay vòng

Khi chuỗi ngữ cảnh bị phân mảnh dọc trên 4 GPU:
*   GPU 0 giữ các tokens từ $0$ đến $250K$.
*   GPU 1 giữ các tokens từ $250K$ đến $500K$.
*   GPU 2 giữ các tokens từ $500K$ đến $750K$.
*   GPU 3 giữ các tokens từ $750K$ đến $1M$.

Để GPU 0 có thể tính toán Attention chính xác cho Query của nó, nó bắt buộc phải so khớp với các Keys và Values nằm trên toàn bộ chuỗi $0 - 1M$ (tức là cần dữ liệu KV nằm trên GPU 1, 2, 3). Giải thuật **Ring Attention** giải quyết việc này bằng cách truyền tin xoay vòng (Ring communication):

```
┌──────────────┐     All-Gather     ┌──────────────┐
│  [ GPU 0 ]   │ ─────────────────> │  [ GPU 1 ]   │
│ Query: 0-250K│                    │ Query: 250K- │
└──────────────┘                    └──────────────┘
       ▲                                   │
       │                                   │ All-Gather
       │ All-Gather                        ▼
┌──────────────┐                    ┌──────────────┐
│  [ GPU 3 ]   │ <───────────────── │  [ GPU 2 ]   │
│ Query: 750K- │                    │ Query: 500K- │
└──────────────┘                    └──────────────┘
```

### Quy trình tính toán xoay vòng:
1.  **Bước khởi đầu**: Mỗi GPU tự tính toán Attention cục bộ giữa Query, Key, Value của chính mình (ví dụ GPU 0 tính Attention của đoạn $0-250K$).
2.  **Xoay vòng dữ liệu**: GPU $i$ truyền Key và Value của mình sang GPU liền kề $i+1$ (GPU 3 truyền sang GPU 0), đồng thời nhận Key và Value mới từ GPU $i-1$. Phép giao tiếp này chạy song song (overlap) hoàn toàn với bước tính toán attention tiếp theo trên GPU.
3.  **Tính toán và Tích lũy**: Mỗi khi nhận được mảnh Key-Value mới từ GPU lân cận, GPU tự động tính toán tích chập attention và tích lũy kết quả trung gian (softmax exponent và log-sum-exp) vào output của mình.
4.  **Kết thúc**: Sau $N-1$ bước xoay (với $N$ là số lượng GPU trong nhóm CP), toàn bộ dữ liệu Key-Value đã đi qua tất cả các GPU. Kết quả Attention cuối cùng trên mỗi GPU đạt độ chính xác tuyệt đối như khi chạy trên bộ nhớ phẳng của GPU đơn lẻ.

---

## 4. Liên hệ với Toy Engine: Sự đơn giản hóa trong mô phỏng

Để hiểu sự phức tạp của song song ngữ cảnh, hãy đối chiếu với mô hình mô phỏng [Toy Serving Engine](file:///Users/admin/TuanDung/vllm-architecture-lectures/toy_engine/) của chúng ta:

*   **Toy Engine (Bộ nhớ phẳng)**: Trong [allocator.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/toy_engine/allocator.py) (`BlockAllocator`) và [scheduler.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/toy_engine/scheduler.py), chúng ta quản lý mỗi request như một thực thể đơn nhất. Chuỗi tokens lịch sử và KV cache của request được lưu trữ trên một danh sách phẳng (`block_table`) chạy trong không gian bộ nhớ RAM duy nhất của tiến trình Python.
*   **Production vLLM (Mảnh bộ nhớ phân tán)**: Trong thực tế vLLM chạy Context Parallelism, KV Cache vật lý của một request bị xé lẻ thành các mảnh độc lập (`dcp_local_seq_lens`) nằm trên các GPU khác nhau. Bộ lập lịch (Scheduler) phải phối hợp cực kỳ đồng bộ: khi chạy tính toán bước Attention, tất cả các worker GPU phải khởi chạy Ring Attention đồng thời để xoay vòng các tensor Key-Value qua mạng liên-GPU, đòi hỏi băng thông cực lớn và sự chính xác tuyệt đối ở mức mili-giây.

---

## 💡 Tổng kết bài học

*   **Tensor Parallelism (TP)** không giải quyết được bài toán ngữ cảnh siêu dài (100K - 1M+ tokens) vì độ dài chuỗi $L$ của KV cache vẫn giữ nguyên trên tất cả các GPU.
*   **Context Parallelism (CP / DCP)** phân chia chính chiều dài chuỗi ngữ cảnh (Sequence Dimension) trên các GPU, được hiện thực hóa qua thuật toán chia xen kẽ round-robin trong [cp_utils.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/worker/gpu/cp_utils.py).
*   **Ring Attention** giải quyết bài toán Attention trên chuỗi ngữ cảnh phân tán bằng cách truyền tin xoay vòng (ring transfer) các tensor Key, Value chéo giữa các GPU để tính toán và cập nhật kết quả tích lũy song song.
