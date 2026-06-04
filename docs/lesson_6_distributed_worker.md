---
layout: default
title: "Bài 6: Distributed Executor & GPU Workers"
---

# Bài 6: Deep Dive Codebase – Distributed Executor & GPU Workers

Khi phục vụ các mô hình ngôn ngữ lớn (LLM) có hàng chục hay hàng trăm tỷ tham số (như Llama 3 70B, Mixtral 8x22B), bản thân trọng số mô hình đã vượt quá dung lượng VRAM của một GPU đơn lẻ (ví dụ 80GB của GPU A100/H100). Do đó, hệ thống AI Serving bắt buộc phải hỗ trợ suy luận phân tán (**Distributed Inference**).

Trong bài này, chúng ta sẽ khảo sát cách vLLM thiết kế phân hệ thực thi song song qua lớp `Executor`, vòng đời của `Worker`, quy trình khởi tạo tự động đo bộ nhớ (`Memory Profiling`), và kỹ thuật tăng tốc cực kỳ quan trọng: **CUDA Graphs**.

---

## 1. Cơ chế Song song hóa: Tensor Parallel vs Pipeline Parallel

Để chạy mô hình trên nhiều GPU, vLLM hỗ trợ hai cơ chế song song hóa chính:

```
1. Tensor Parallelism (Song song hóa Tensor - TP):
Prompt ---> [ GPU 0: Phân nửa Ma trận W1 ] ---\ NCCL All-Reduce
      ---> [ GPU 1: Phân nửa Ma trận W1 ] ---/ (Truyền thông tin đồng bộ nhanh)

2. Pipeline Parallelism (Song song hóa Đường ống - PP):
Prompt ---> [ GPU 0: Các tầng 1-16 ] ---> [ GPU 1: Các tầng 17-32 ] ---> Output
```

### Tensor Parallelism (TP):
* **Nguyên lý**: Chia nhỏ các phép nhân ma trận trọng số (ví dụ: các tầng MLP hay Attention) của mô hình ra làm nhiều phần (theo cột hoặc theo dòng) để tính toán song song trên nhiều GPU cùng một lúc.
* **Đặc điểm**: Đòi hỏi giao tiếp mạng cực kỳ nhanh giữa các GPU ở mỗi tầng của mô hình (thông qua phép toán `All-Reduce` của thư viện NCCL). TP hoạt động tốt nhất khi các GPU nằm trên cùng một node vật lý và kết nối với nhau qua cầu nối tốc độ cao NVLink.
* **vLLM Integration**: vLLM sử dụng thư viện Megatron-LM style để song song hóa mô hình PyTorch một cách mượt mà.

### Pipeline Parallelism (PP):
* **Nguyên lý**: Chia mô hình theo chiều dọc (ví dụ: GPU 0 giữ tầng 1-16, GPU 1 giữ tầng 17-32). Dữ liệu tính toán xong ở GPU 0 sẽ được truyền phát sang GPU 1 qua mạng.
* **vLLM Integration**: vLLM v1 tối ưu hóa PP bằng cách cho phép xếp hàng nhiều batch (`batch_queue_size > 1`) chạy đan xen để loại bỏ khoảng thời gian GPU bị rảnh (pipeline bubbles).

---

## 2. Vòng đời và Phân cấp điều khiển: Executor & Workers

Để quản lý nhiều tiến trình/GPU, vLLM định nghĩa cấu trúc phân cấp:

```
         [ EngineCore ]
               |
        [ GPU Executor ]  (Ray / Multiprocessing)
         /            \
 [ GPU 0: Worker ]   [ GPU 1: Worker ]
        |                   |
 [ ModelRunner ]     [ ModelRunner ]
```

### GPU Executor:
Tùy thuộc vào môi trường chạy, vLLM sử dụng các Executor khác nhau (kế thừa từ `Executor` cơ sở):
* **`RayGPUExecutor`**: Sử dụng framework Ray để quản lý các GPU vật lý, đặc biệt hữu ích khi chạy phân tán trên cụm máy chủ gồm nhiều node vật lý khác nhau.
* **`MultiprocessingGPUExecutor`**: Sử dụng tiến trình con Python (`multiprocessing`) để quản lý các GPU nằm trên cùng một máy chủ đơn lẻ, giảm thiểu overhead quản lý của Ray.

### Worker:
Mỗi GPU vật lý được kiểm soát bởi một tiến trình **`Worker`** riêng biệt. Nhiệm vụ chính của `Worker` là:
* Khởi tạo thư viện PyTorch, NCCL và Ray/Multiprocessing group.
* Khởi tạo lớp **`ModelRunner`**: Nơi chứa mô hình PyTorch thực tế và trực tiếp chạy các tính toán forward.

---

## 3. Khởi tạo & Quy trình Memory Profiling (Warmup)

Trước khi bắt đầu nhận request của người dùng, vLLM phải tính toán xem có bao nhiêu khối vật lý (Physical Blocks) có thể cấp phát cho KV Cache. Việc tính toán này được thực hiện tự động qua quy trình **Memory Profiling**:

```
Bước 1: Worker nạp trọng số mô hình (Weights) vào GPU VRAM.
Bước 2: Chạy thử mô hình với một batch giả lập (Dummy/Warmup execution) ở kích thước lớn nhất.
Bước 3: Đo lượng VRAM còn trống (Available Memory) sau khi trừ đi VRAM chứa Weights, CUDA context và bộ nhớ đệm tính toán.
Bước 4: Tính số lượng Blocks vật lý tối đa:
        num_blocks = Available Memory / Block Size (in Bytes)
Bước 5: Cấu hình VLLM Block Pool với số lượng block vật lý này.
```

### Cách tính toán kích thước của 1 Khối KV Cache (Block Size in Bytes):
Ví dụ với Llama 3 8B (FP16, block_size = 16, GQA với 8 đầu KV, head_dim = 128, 32 tầng):

$$\text{Bytes per Block} = 2 \times 32 \text{ layers} \times 8 \text{ heads} \times 128 \text{ head\_dim} \times 2 \text{ Bytes} \times 16 \text{ tokens} = 2,097,152 \text{ Bytes} = 2\text{ MB}$$

Nếu sau khi warmup, GPU còn trống $20\text{ GB}$ VRAM:

$$\text{Num Blocks} = \frac{20 \times 1024\text{ MB}}{2\text{ MB}} = 10,240 \text{ Blocks}$$

Số lượng block vật lý này sẽ được báo ngược về cho bộ lập lịch `Scheduler` để bắt đầu quản lý.

---

## 4. Tăng tốc tối đa: CUDA Graphs Capture

Trong pha sinh token (Decode Phase), kích thước batch nhỏ (GEMV) làm cho thời gian thực thi của mỗi kernel tính toán trên GPU diễn ra cực kỳ nhanh (dưới 1ms). 
* **Vấn đề**: Việc gọi liên tục hàng chục kernel PyTorch từ phía CPU thông qua mã Python tốn rất nhiều thời gian chuẩn bị lệnh. Độ trễ do CPU chuẩn bị lệnh và kích hoạt kernel (Kernel Launch Overhead) có thể chiếm tới **50% đến 70%** tổng thời gian sinh token.
* **Giải pháp: CUDA Graphs**:
  * CUDA Graphs cho phép ghi lại (capture) toàn bộ chuỗi các cuộc gọi kernel GPU, các tham số đầu vào và luồng phụ thuộc dữ liệu thành một biểu đồ đồ thị tĩnh (Graph) trong bộ nhớ GPU.
  * Trong các bước chạy Decode tiếp theo, thay vì Python phải gọi từng kernel một, hệ thống chỉ cần ra lệnh thực thi đúng 1 biểu đồ CUDA Graph duy nhất. Quá trình launch này xảy ra hoàn toàn trên GPU, loại bỏ 100% CPU overhead.

```
Không dùng CUDA Graphs:
CPU: [ Launch K1 ] ---> [ Launch K2 ] ---> [ Launch K3 ] ---> (Lặp lại ở mỗi Token)
GPU:   [ Run K1 ]         [ Run K2 ]         [ Run K3 ]

Sử dụng CUDA Graphs:
CPU: [ Launch 1 Graph ] -------------------------------------> (Chỉ 1 lệnh launch)
GPU:   [ Run K1 -> Run K2 -> Run K3 ] (Chạy liên tục siêu nhanh)
```

### Cách vLLM hiện thực CUDA Graphs:
* Vì CUDA Graph yêu cầu địa chỉ con trỏ tensor đầu vào và đầu ra phải cố định trong bộ nhớ GPU, vLLM sử dụng cơ chế lưu trữ KV Cache dạng phân trang (PagedAttention) có địa chỉ các khối vật lý cố định.
* vLLM sẽ capture trước một số lượng đồ thị CUDA Graphs tương ứng với các kích thước Batch size khác nhau (ví dụ: Batch size = 1, 2, 4, 8, 16, 32...). Khi chạy, tùy thuộc vào số lượng request được lập lịch ở bước đó, `ModelRunner` sẽ chọn đồ thị CUDA Graph phù hợp nhất để thực thi.

---

## 💡 Tổng kết bài học
* vLLM sử dụng **Tensor Parallelism** (chia nhỏ ma trận qua NCCL) và **Pipeline Parallelism** để chạy các mô hình lớn vượt quá dung lượng 1 GPU.
* **Memory Profiling** là bước chạy dummy ở pha khởi động để tự động xác định không gian VRAM khả dụng tối đa cho KV Cache Block Pool.
* **CUDA Graphs** giúp loại bỏ độ trễ của CPU khi launch các GPU kernel lặp đi lặp lại trong pha Decode, đem lại tốc độ sinh token tối đa.

Trong bài học tiếp theo, chúng ta sẽ khảo sát các kỹ thuật tối ưu hóa nâng cao khác như **Speculative Decoding, Multi-LoRA, và Lượng hóa (Quantization)**.
