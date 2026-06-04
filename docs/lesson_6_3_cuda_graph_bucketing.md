---
sidebar_position: 8.8
sidebar_label: "Bài 6.3: CUDA Graph & Shape Bucketing"
---

# Bài 6.3: CUDA Graph Capture & Cơ chế Shape Bucketing

Trong pha giải mã sinh token (**Decode Phase**) của LLM, hệ thống phục vụ phải liên tục thực hiện các bước suy luận với độ dài đầu vào cực ngắn ($N = 1$ token cho mỗi request). Về mặt tính toán, GPU hoàn tất việc thực thi các kernel tính toán cho 1 token chỉ trong vòng vài chục micro-giây.

Tuy nhiên, việc gọi liên tiếp hàng trăm kernel PyTorch từ phía CPU thông qua mã Python tốn rất nhiều thời gian chuẩn bị lệnh. Hiện tượng này được gọi là **CPU Execution Overhead**. 

Bài học này sẽ phân tích cách vLLM giải phóng CPU khỏi nút thắt cổ chai này nhờ cơ chế **CUDA Graphs** kết hợp giải thuật **Shape Bucketing**.

---

## 1. Nút thắt cổ chai: Kernel Launch Overhead trên CPU

Khi thực thi mô hình LLM (ví dụ: Llama 3 8B với 32 layers, mỗi layer chứa khoảng 10-15 phép toán/kernels độc lập):
*   Số lượng GPU kernels cần chạy cho mỗi token ước tính khoảng $32 \times 12 = 384$ kernels.
*   **Chi phí kích hoạt (Kernel Launch Overhead)**: Mỗi lần CPU gọi một hàm CUDA từ Python sang driver GPU, driver mất khoảng $5 - 15\text{ }\mu\text{s}$ để dịch lệnh và đẩy xuống hàng đợi phần cứng.
*   **Tổng chi phí CPU**:
    $$\text{Overhead}_{\text{launch}} \approx 384 \times 10\text{ }\mu\text{s} = 3.84\text{ ms}$$
*   Nếu thời gian thực tính toán của GPU cho 1 bước Decode chỉ mất $3\text{ ms}$: **CPU lúc này chiếm tới hơn 55% tổng thời gian suy luận**, giữ GPU ở trạng thái rảnh rỗi chờ lệnh từ CPU. GPU có mạnh đến đâu cũng bị nghẽn bởi tốc độ luồng xử lý đơn nhân của CPU.

---

## 2. Giải pháp: CUDA Graphs

**CUDA Graphs** (được NVIDIA giới thiệu từ CUDA 10) thay đổi hoàn toàn cách CPU tương tác với GPU. Thay vì gửi từng lệnh đơn lẻ, CUDA Graphs cho phép ghi lại (capture) toàn bộ chuỗi các cuộc gọi kernel, cấu hình lưới (grid/block dimensions) và luồng phụ thuộc dữ liệu thành một **đồ thị tĩnh (Static Graph)** duy nhất trong bộ nhớ GPU.

```
Không dùng CUDA Graphs:
CPU: [ Dịch K1 ] ➔ [ Launch K1 ] ➔ [ Dịch K2 ] ➔ [ Launch K2 ] ...
GPU:               [ Chạy K1 ]                    [ Chạy K2 ]
(* GPU liên tục bị ngắt quãng do đợi CPU launch)

Sử dụng CUDA Graphs:
Pha Capture (Chỉ làm 1 lần lúc khởi động):
CPU: [ Ghi lại toàn bộ chuỗi K1 -> K2 -> ... -> Kn ] ➔ Tạo Graph trên GPU

Pha Thực thi (Run ở mỗi bước Decode):
CPU: [ Gọi 1 lệnh Run Graph ] ------------------------┐
GPU:                                                 ▼
                                     [ Chạy K1 ➔ Chạy K2 ➔ ... ➔ Chạy Kn ]
(* GPU chạy liên tục không bị ngắt quãng bởi CPU)
```

### Giới hạn vật lý của CUDA Graphs:
CUDA Graph yêu cầu tất cả các thuộc tính của các phép toán trong đồ thị phải **cố định**:
1.  **Địa chỉ con trỏ bộ nhớ (Memory Pointers)** của các tensor đầu vào, đầu ra và trung gian không được thay đổi.
2.  **Hình dáng của Tensor (Tensor Shapes)** (như batch size, chiều dài ngữ cảnh) phải hoàn toàn cố định.

Nếu thay đổi shape (ví dụ: số lượng request trong batch tăng từ 3 lên 5), đồ thị cũ sẽ bị vô hiệu hóa và ta bắt buộc phải capture lại một đồ thị mới (tốn vài giây, gây trễ cực lớn).

---

## 3. Cách vLLM vượt qua giới hạn: PagedAttention & Shape Bucketing

vLLM kết hợp cấu trúc phân trang độc đáo của mình để biến CUDA Graphs thành hiện thực trong serving động:

### 3.1. Cố định địa chỉ nhớ nhờ Paged KV Cache
Nhờ PagedAttention, địa chỉ bộ nhớ của các khối vật lý chứa KV Cache trên GPU đã được cấp phát cố định sẵn trong Block Pool ngay từ lúc khởi động hệ thống. Khi chạy suy luận, vLLM chỉ tráo đổi các chỉ mục (indices) trong bảng trang (Page Table), hoàn toàn không thay đổi địa chỉ con trỏ tensor vật lý của bộ nhớ cache.

### 3.2. Cơ chế Shape Bucketing (Gom nhóm kích thước)
Để xử lý sự biến động động của kích thước batch (Batch Size) tại mỗi bước Continuous Batching, vLLM sử dụng cơ chế **Shape Bucketing**:

*   vLLM định nghĩa trước một danh sách các kích thước Batch size tiêu chuẩn được hỗ trợ cho đồ thị (được gọi là **Buckets / Cuda Graph Sizes**). Ví dụ:
    $$\text{Buckets} = \{1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128\}$$
*   **Pha Khởi động (Warmup/Capture Phase)**:
    *   vLLM thực hiện chạy giả lập (dummy runs) và capture sẵn các đồ thị CUDA Graphs tương ứng với từng kích thước batch trong tập hợp trên.
    *   Tất cả các đồ thị này được lưu trữ sẵn trong VRAM.
*   **Pha Lập lịch thực tế**:
    *   Giả sử tại một bước lập lịch, bộ lập lịch chọn ra được **5 request** đang chạy Decode (Batch size thực tế $B = 5$).
    *   vLLM sẽ chọn đồ thị CUDA Graph của bucket lớn hơn gần nhất: **Bucket $B_{\text{padded}} = 8$**.
    *   Hệ thống thực hiện chèn thêm 3 request giả lập (Dummy Inputs - padding bằng các token rác) để làm đầy lô lên kích thước 8.
    *   Chạy đồ thị CUDA Graph 8. Kết quả đầu ra của 3 request giả lập sẽ bị loại bỏ ở pha Sampler.

```
Batch thực tế (B = 5): [ Req 1, Req 2, Req 3, Req 4, Req 5 ]
         │
         ▼ (Đệm thêm 3 request ảo)
Batch padded  (B = 8): [ Req 1, Req 2, Req 3, Req 4, Req 5, PAD, PAD, PAD ]
         │
         ▼ (Gọi đồ thị đã capture sẵn cho Batch 8)
[ Thực thi CUDA Graph 8 ] ➔ Thu được 8 logits ➔ Lọc lấy 5 logits thực tế.
```

---

## 4. Sự đánh đổi: Memory Overhead của CUDA Graphs

Mặc dù mang lại tốc độ vượt trội (giảm ITL từ 1.5x đến 2x), CUDA Graphs đòi hỏi chi phí lưu trữ VRAM rất lớn:

*   Mỗi đồ thị CUDA Graph khi capture sẽ tạo ra một vùng nhớ đệm tĩnh (Workspace Memory Pool) để chứa toàn bộ các tensor kích hoạt trung gian của tất cả các lớp trong mô hình.
*   Dung lượng bộ nhớ này tỷ lệ thuận với kích thước mô hình và kích thước Batch size của bucket.
*   Nếu chúng ta capture quá nhiều bucket hoặc kích thước bucket quá lớn, vùng nhớ dành cho CUDA Graph sẽ nuốt chửng hàng chục Gigabytes VRAM, làm giảm đáng kể số lượng block vật lý còn lại cho KV Cache, gián tiếp hạ thấp tổng Throughput của hệ thống.

Do đó, vLLM mặc định chỉ bật CUDA Graphs cho pha Decode (vốn có shape cố định $N = 1$ ở chiều sequence), và giới hạn kích thước batch tối đa được capture (thường mặc định là `--max-num-seqs=256`).

---

## 5. Khảo sát mã nguồn thực tế trong vLLM v1

Trong phiên bản vLLM v1, logic capture và quản lý đồ thị được thực hiện thông qua lớp `ModelCudaGraphManager` phối hợp với `model_runner.py`:

*   Tại [vllm/v1/worker/gpu/model_runner.py](file:///Users/admin/TuanDung/repos/vllm/vllm/v1/worker/gpu/model_runner.py), hàm `initialize_kv_cache` khởi tạo trình quản lý đồ thị:

```python
self.cudagraph_manager = ModelCudaGraphManager(
    self.vllm_config,
    self.device,
    cudagraph_mode,
    decode_query_len=self.decode_query_len,
)
```

*   Hàm `capture_model` thực thi việc dọn rác và capture toàn bộ đồ thị:

```python
@torch.inference_mode()
def capture_model(self) -> int:
    # 1. Thu gom rác hệ thống và giải phóng bộ đệm CUDA cache
    gc.collect()
    torch.accelerator.empty_cache()
    start_free_gpu_memory = torch.cuda.mem_get_info()[0]
    
    # 2. Gọi trình quản lý đồ thị thực thi capture model
    captured_attn_states = self.cudagraph_manager.capture(
        self.model,
        self.model_state,
        self.input_buffers,
        ...
    )
    
    end_free_gpu_memory = torch.cuda.mem_get_info()[0]
    cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
    return cuda_graph_size # Trả về dung lượng VRAM bị chiếm dụng
```

---

## 💡 Tổng kết bài học

*   **Launch Overhead** trên CPU là rào cản lớn nhất đối với pha Decode có latency siêu ngắn, do CPU mất nhiều thời gian chuẩn bị lệnh cho hàng trăm GPU kernels.
*   **CUDA Graphs** gom toàn bộ luồng kernels thành một đồ thị tĩnh, cho phép CPU kích hoạt toàn bộ luồng tính toán chỉ bằng 1 lệnh launch duy nhất.
*   **Shape Bucketing** là giải thuật đệm đắp (padding) batch size thực tế lên các kích thước đệm cố định đã được capture sẵn để tương thích với giới hạn hình dáng tĩnh của CUDA Graphs.
*   Việc bật CUDA Graphs yêu cầu đánh đổi một phần **VRAM bộ nhớ đệm** tĩnh, đổi lại tốc độ phản hồi inter-token nhanh gấp đôi trong môi trường serving thực tế.
