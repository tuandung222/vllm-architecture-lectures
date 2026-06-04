---
sidebar_position: 11.8
sidebar_label: "Bài 7.4: Phân rã Prefill & Decode"
---

# Bài 7.4: Kiến trúc Phân rã Prefill & Decode (Prefill-Decode Disaggregation)

Trong các bài học trước, chúng ta đã tìm hiểu nhiều kỹ thuật để tối ưu hóa sự xung đột giữa pha **Prefill** (xử lý prompt - Compute-bound) và pha **Decode** (sinh token - Memory-bound) khi chúng chạy chung trên cùng một GPU (như Chunked Prefill ở Bài 3.2). 

Tuy nhiên, khi quy mô phục vụ lên tới hàng triệu người dùng và chiều dài ngữ cảnh tăng lên hàng chục ngàn tokens, giải pháp chia sẻ chung GPU bắt đầu chạm giới hạn vật lý. Để giải quyết triệt để, các hệ thống serving hiện đại nhất (bao gồm cả vLLM) đang dịch chuyển sang một kiến trúc đột phá: **Phân rã Prefill & Decode (Prefill-Decode Disaggregation)**.

Bài học này sẽ mổ xẻ lý thuyết thiết kế hệ thống và luồng dữ liệu của kiến trúc phân tán tiên tiến này.

---

## 1. Bản chất xung đột phần cứng giữa Prefill và Decode

Tại sao việc chạy chung Prefill và Decode trên cùng một GPU lại không tối ưu về mặt vật lý phần cứng?

1.  **Sự lệch pha về tài nguyên (Resource Discrepancy)**:
    *   **Prefill**: Cần năng lực tính toán cực lớn (FLOPs) của các **Tensor Cores** để thực hiện nhân ma trận lớn (GEMM). Băng thông bộ nhớ không phải là nút thắt cổ chai chính ở pha này.
    *   **Decode**: Không cần nhiều FLOPs, nhưng đòi hỏi **băng thông bộ nhớ cực cao (High VRAM Bandwidth)** để nạp trọng số mô hình và KV Cache liên tục từ HBM sang SRAM cho phép tính GEMV.
2.  **Hiện tượng Nghẽn hàng đợi (Queueing Delay & Interference)**:
    *   Khi một prefill lớn đang chạy, nó độc chiếm các SMs (Streaming Multiprocessors) của GPU.
    *   Các decode step (vốn rất ngắn) phải xếp hàng đợi prefill xong mới được thực thi. Điều này làm tăng độ trễ **Inter-Token Latency (ITL)** và gây hiện tượng giật cục hiển thị văn bản (jitter).

---

## 2. Kiến trúc Phân rã (Disaggregated Serving Architecture)

Cơ chế phân rã giải quyết xung đột bằng cách chia hệ thống thành hai cụm máy chủ phần cứng chuyên biệt, hoạt động độc lập:

```
                  ┌───────────────────────────────┐
                  │      Global Router / LB       │
                  └──────────────┬────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼ (1. Gửi Prompt)           │ (4. Chuyển tiếp Request)
         ┌───────────────────┐                 ▼
         │   Prefill Node    │       ┌───────────────────┐
         │ (Compute-Optimized│       │    Decode Node    │
         │   e.g. H100 GPU)  │       │ (Memory-Optimized │
         └─────────┬─────────┘       │   e.g. A100 GPU)  │
                   │                 └─────────▲─────────┘
                   │                           │
                   └───────────(2. KV Cache)───┘
                               (GPUDirect RDMA)
```

### Luồng đi của một Request (Request Lifecycle):
1.  **Nhận Request**: Request mới (Prompt) đi vào bộ định tuyến trung tâm (Global Router).
2.  **Xử lý Prefill**: Router gửi request tới cụm **Prefill Nodes** (Sử dụng các GPU mạnh về compute như H100). Prefill Node chạy song song toàn bộ prompt, sinh ra các token đầu tiên và tính toán xong **KV Cache** của prompt đó.
3.  **Truyền tải KV Cache (KV Cache Transfer)**: Prefill Node đóng gói và gửi toàn bộ KV Cache vừa tính toán được qua mạng tốc độ cao sang cụm **Decode Nodes**.
4.  **Xử lý Decode**: Decode Node (Sử dụng các GPU tối ưu về băng thông bộ nhớ hoặc dung lượng VRAM lớn) nhận KV Cache, nạp vào bảng trang của mình và tiếp tục chạy pha Decode sinh các token tiếp theo một cách độc lập cho đến khi hoàn thành.

---

## 3. Thử thách kỹ thuật: Nút thắt truyền tải KV Cache qua mạng

Thử thách lớn nhất của kiến trúc này là **độ trễ truyền tải KV Cache qua mạng**. Nếu thời gian truyền KV Cache từ cụm Prefill sang cụm Decode lớn hơn thời gian chạy Prefill trực tiếp trên GPU, kiến trúc này sẽ hoàn toàn thất bại.

### 3.1. Tính toán dung lượng KV Cache cần truyền tải:
Ví dụ với mô hình Llama 3 8B (FP16, GQA, 32 layers, 8 KV heads, head_dim = 128) với prompt dài **4096 tokens**:

$$\text{KV Cache Size} = 2 \times 32 \text{ layers} \times 8 \text{ heads} \times 128 \text{ head\_dim} \times 2 \text{ Bytes} \times 4096 \text{ tokens} = 536,870,912 \text{ Bytes} \approx 512 \text{ MB}$$

*   Nếu truyền qua mạng LAN thông thường $10\text{ Gbps}$ (tốc độ thực tế $\approx 1\text{ GB/s}$): Mất **500ms** chỉ để truyền cache. Điều này phá hỏng hoàn toàn TTFT.

### 3.2. Giải pháp kỹ thuật (GPUDirect RDMA)
Để đạt được thời gian truyền dưới 10ms, hệ thống serving sử dụng:
1.  **Mạng RDMA (Remote Direct Memory Access)**: Sử dụng các giao thức RoCEv2 hoặc InfiniBand với băng thông cực cao ($200\text{ Gbps}$ đến $400\text{ Gbps}$, tương đương $25 - 50\text{ GB/s}$).
2.  **GPUDirect RDMA**: Cho phép card mạng (NIC) đọc trực tiếp dữ liệu từ VRAM của GPU trên Prefill Node và ghi trực tiếp vào VRAM của GPU trên Decode Node thông qua PCIe/NVLink switch, **hoàn toàn bypass qua CPU RAM và Kernel của OS**.
    *   Với tốc độ $400\text{ Gbps}$ ($50\text{ GB/s}$), việc truyền tải $512\text{ MB}$ KV Cache chỉ mất:
        $$\text{Time}_{\text{transfer}} = \frac{512\text{ MB}}{50\text{ GB/s}} \approx 10\text{ ms}$$
    *   Khoảng thời gian 10ms này là hoàn toàn chấp nhận được và bị lấn át bởi lợi ích giảm ITL cực lớn ở cụm Decode.

---

## 4. Phối hợp với RadixAttention (Prefix Caching disaggregated)

Một tối ưu hóa đỉnh cao khác là kết hợp **RadixAttention (Bài 2.2)** vào mô hình phân rã:
*   Decode Nodes cũng duy trì cấu trúc cây Radix Tree lưu trữ KV Cache của các prompt hệ thống hoặc tài liệu phổ biến.
*   Khi Prefill Node hoàn thành xử lý một prompt hệ thống dùng chung, thay vì gửi toàn bộ KV Cache qua mạng, nó chỉ cần gửi một mã băm (Hash Key) cho Decode Node.
*   Nếu Decode Node phát hiện ra mình đã có sẵn KV Cache của tiền tố này trong Radix Tree của mình (**Cache Hit** cục bộ), nó sẽ sử dụng luôn bản cache đó và bỏ qua hoàn toàn việc truyền tải KV Cache dung lượng lớn qua mạng!

---

## 💡 Tổng kết bài học

*   **Prefill-Decode Disaggregation** giải quyết triệt để sự xung đột về tài nguyên SMs giữa pha Prefill (Compute-bound GEMM) và Decode (Memory-bound GEMV) bằng cách tách biệt chúng trên các GPU khác nhau.
*   Mô hình này giúp **cố định và tối ưu hóa độ trễ sinh token (ITL)** cho người dùng đang nhận stream, không bị gián đoạn bởi các prompt mới cực dài đi vào hệ thống.
*   **GPUDirect RDMA** là xương sống công nghệ giúp truyền tải KV Cache dung lượng lớn giữa các GPU liên node thông qua mạng InfiniBand/RoCE mà không bị nghẽn bởi CPU hay OS kernel.
*   Sự kết hợp giữa disaggregated serving và **RadixAttention** giúp giảm thiểu dung lượng truyền tải mạng nhờ tái sử dụng KV cache tiền tố có sẵn trên cụm Decode.
