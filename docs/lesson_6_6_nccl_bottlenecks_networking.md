---
sidebar_position: 8.95
sidebar_label: "Bài 6.6: NCCL Bottlenecks & Tối ưu hóa mạng"
---

# Bài 6.6: NCCL Bottlenecks & Tối ưu hóa mạng phục vụ

Khi triển khai mô hình ngôn ngữ lớn (LLM) trên môi trường đa GPU, hiệu năng phục vụ thực tế của hệ thống thường không bị giới hạn bởi tốc độ tính toán Tensor Core của GPU, mà bị bóp nghẹt bởi **băng thông giao tiếp mạng giữa các GPU**.

Bài học này sẽ mổ xẻ tại sao pha Decode của Transformer lại cực kỳ nhạy cảm với độ trễ truyền thông, phân tích sự khác biệt vật lý giữa kết nối NVLink và PCIe, và cung cấp cẩm nang thực chiến tinh chỉnh các biến môi trường **NCCL (NVIDIA Collective Communications Library)** để tối ưu hóa băng thông mạng phục vụ.

---

## 1. Bản chất của nghẽn giao tiếp NCCL ở pha Decode (Communication-bound)

Để hiểu rõ nguyên nhân gây nghẽn mạng, chúng ta phải phân biệt hành vi giao tiếp của GPU giữa hai pha: **Prefill** và **Decode**.

### Pha Prefill: NCCL dung lượng lớn (Thích hợp cho GPU)
Trong pha Prefill, GPU xử lý song song toàn bộ prompt đầu vào. Lượng tính toán GEMM cực kỳ lớn (Compute-bound). Giao tiếp NCCL All-Reduce giữa các GPU tuy truyền tải các tensor kích thước lớn nhưng tần suất gọi rất ít (chỉ gọi 1 lần duy nhất cho mỗi layer tại pha prefill). Thời gian truyền thông NCCL dễ dàng được che giấu (overlap) đằng sau thời gian tính toán khổng lồ của GPU.

### Pha Decode: NCCL tần suất cao, dung lượng siêu nhỏ (Ác mộng của mạng)
Trong pha Decode, GPU chỉ xử lý đúng **1 token mới** ở mỗi bước. Lượng tính toán ở mỗi bước cực kỳ nhỏ và kết thúc rất nhanh. Tuy nhiên:
*   Mỗi bước sinh token bắt buộc phải chạy All-Reduce để đồng bộ hóa kích hoạt (activations) sau mỗi tầng Attention (và MLP nếu dùng Tensor Parallelism).
*   Với mô hình Llama-3-70B (80 layers) chạy TP=4: Mỗi bước sinh 1 token duy nhất yêu cầu hệ thống phải thực thi **80 lần All-Reduce liên tiếp** trên cả 4 GPU.
*   Lượng dữ liệu truyền ở mỗi lần All-Reduce chỉ là vector kích hoạt của 1 token duy nhất (chỉ vài chục KB).

Điều này khiến pha Decode rơi hoàn toàn vào trạng thái **Communication-bound (Nghẽn truyền thông)**. Lúc này, độ trễ khởi động (kernel launch latency) của thư viện NCCL và độ trễ vật lý của đường truyền mạng giữa các GPU trở thành yếu tố quyết định tốc độ sinh token (Inter-Token Latency), chứ không phải năng lực tính toán của GPU.

---

## 2. NVLink vs PCIe: Sự khác biệt vật lý quyết định hiệu năng

Khi chạy Tensor Parallelism (TP), việc các GPU kết nối với nhau qua giao tiếp nào sẽ trực tiếp quyết định speculative decoding hay decode thông thường chạy nhanh hay chậm.

| Đặc tính | PCIe Gen4 x16 | PCIe Gen5 x16 | NVLink (A100 / H100) |
| :--- | :--- | :--- | :--- |
| **Băng thông lý thuyết** | 32 GB/s | 64 GB/s | 600 GB/s - 900 GB/s trở lên |
| **Băng thông thực tế** | ~26 GB/s | ~50 GB/s | > 500 GB/s |
| **Độ trễ truyền tải** | Cao (đi qua CPU Root Complex) | Trung bình | Cực thấp (giao tiếp trực tiếp GPU-to-GPU) |

### Tại sao chạy Multi-GPU không có NVLink sẽ làm giảm hiệu năng?
Khi không có kết nối NVLink (ví dụ khi thuê các máy ảo GPU giá rẻ trên Cloud kết nối qua khe cắm PCIe thông thường):
1.  **PCIe Bottleneck**: Ở mỗi layer trong số 80 layers của mô hình, dữ liệu activations phải được gửi từ GPU này sang GPU khác đi qua chip cầu PCIe và CPU. Băng thông PCIe quá hẹp và độ trễ cao khiến GPU phải đứng rảnh rỗi chờ đợi nhận dữ liệu.
2.  **Hiệu năng đảo chiều**: Tốc độ sinh token khi chạy TP=2 hoặc TP=4 trên các GPU kết nối qua PCIe có thể **chậm hơn gấp đôi** so với việc chạy mô hình đã được lượng hóa (quantized) trên đúng 1 GPU duy nhất không cần chia nhỏ!

> [!IMPORTANT]
> **Quy tắc vàng trong Production:**
> Chỉ sử dụng cấu hình Tensor Parallelism (TP) khi và chỉ khi các GPU trên server vật lý được liên kết trực tiếp bằng kết nối **NVLink** tốc độ cao. Nếu không có NVLink, hãy chuyển hướng sang sử dụng **Pipeline Parallelism (PP)** hoặc lượng hóa mô hình (AWQ, FP8) để nhét vừa mô hình vào 1 GPU đơn lẻ.

---

## 3. NVLink-Sharded KV Cache

Để giảm thiểu tối đa overhead truyền dữ liệu KV Cache giữa các GPU, vLLM áp dụng kiến trúc sharded KV cache thông minh trên nền NVLink:

```
[ GPU 0 (KV Cache Part 0) ] ──┐
[ GPU 1 (KV Cache Part 1) ] ──┼── (Truy xuất song song qua NVLink)
[ GPU 2 (KV Cache Part 2) ] ──┤
[ GPU 3 (KV Cache Part 3) ] ──┘
```

*   Mỗi GPU chỉ lưu trữ một phần (shard) của KV Cache tương ứng với các đầu Attention (Attention Heads) được phân bổ cho GPU đó theo cơ chế Tensor Parallelism.
*   Khi chạy tính toán Attention, thay vì gom toàn bộ KV Cache về một GPU (gây nghẽn mạng), mỗi GPU tự tính toán cục bộ phần attention của mình trên shard cache tương ứng.
*   Chỉ có kết quả đầu ra (Attention Output) kích thước nhỏ mới được đồng bộ chéo giữa các GPU qua NVLink bằng All-Reduce ở cuối layer. Cơ chế này tận dụng tối đa băng thông NVLink và loại bỏ 95% lượng dữ liệu cần truyền tải trên mạng.

---

## 4. Cẩm nang tinh chỉnh biến môi trường NCCL (NCCL Tuning)

Để tối ưu hóa giao tiếp mạng liên-GPU trên môi trường production, bạn có thể tinh chỉnh các biến môi trường của NCCL trước khi chạy lệnh khởi động vLLM:

### 4.1. Bật gỡ lỗi mạng (`NCCL_DEBUG`)
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL
```
*   *Tác dụng*: Ép NCCL in ra toàn bộ thông tin chi tiết về topology mạng của server khi khởi chạy. Bạn sẽ kiểm tra được chính xác các GPU đang kết nối qua NVLink (`NVLink`) hay bị rơi về PCIe (`SYS` / `PHB`). Nếu thấy log báo dùng PCIe, bạn cần kiểm tra lại phần cứng.

### 4.2. Khống chế số lượng kết nối CUDA (`CUDA_DEVICE_MAX_CONNECTIONS`)
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```
*   *Tác dụng*: Giới hạn mỗi thiết bị GPU chỉ tạo tối đa 1 luồng kết nối đồng thời để tránh tranh chấp tài nguyên mạng. Giúp đồng bộ hóa luồng tính toán của GPU và luồng truyền thông NCCL, tránh hiện tượng đụng độ gây lag packet truyền tin trong quá trình decode đa luồng.

### 4.3. Cấu hình kích thước đệm truyền thông (`NCCL_BUFFSIZE`)
```bash
export NCCL_BUFFSIZE=2097152
```
*   *Tác dụng*: Đặt kích thước buffer của mỗi kênh truyền thông NCCL là 2MB (mặc định thường là 4MB). Đối với pha Decode (các gói tin truyền cực nhỏ), việc giảm kích thước buffer giúp giải phóng bộ nhớ GPU và tăng tốc độ flush dữ liệu trên các kênh truyền, giảm độ trễ truyền tin của các gói nhỏ.

### 4.4. Tắt InfiniBand khi không cần thiết (`NCCL_IB_DISABLE`)
```bash
export NCCL_IB_DISABLE=1
```
*   *Tác dụng*: Nếu bạn chạy trên server đơn node không có card mạng InfiniBand chuyên dụng nhưng driver hệ thống vẫn bật dịch vụ IB, NCCL có thể bị treo do cố gắng tìm kiếm card mạng IB. Việc tắt chủ động giúp NCCL lập tức sử dụng Shared Memory (`shm`) nội bộ để giao tiếp, loại bỏ lỗi timeout khởi chạy.

---

## 5. Liên hệ với Toy Engine: Bỏ qua chi phí truyền thông liên-GPU

Hãy đối chiếu cách xử lý độ trễ trong mô phỏng của chúng ta với thực tế phần cứng:

*   **Toy Engine (Bỏ qua chi phí truyền thông)**: Trong [model.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/toy_engine/model.py), khi mô phỏng thời gian chạy model, chúng ta cộng gộp tất cả độ trễ thành một con số latency giả định. Điều này bỏ qua toàn bộ chi phí vật lý của việc truyền dữ liệu. Chúng ta giả định rằng thông tin được truyền tức thời giữa API Server, Scheduler và GPU với chi phí bằng 0.
*   **Production vLLM (Độ trễ NCCL thực tế)**: Trong thực tế đa GPU, độ trễ sinh token chịu ảnh hưởng nặng nề bởi NCCL communication latency. All-Reduce ở pha Decode truyền tải lượng dữ liệu rất nhỏ nhưng tần suất vô cùng dày đặc. Nếu hệ thống của bạn gặp lỗi cấu hình hoặc không có NVLink, thời gian nghẽn mạng này sẽ trực tiếp cộng thêm vào thời gian forward của GPU, biến một hệ thống lẽ ra phải nhanh hơn thành chậm hơn. Vì vậy, các tinh chỉnh biến môi trường NCCL trong bài học này là chìa khóa để đưa hiệu năng thực tế tiệm cận với hiệu năng lý thuyết mà chúng ta mô phỏng.

---

## 💡 Tổng kết bài học

*   Pha **Decode** thực thi hàng chục lần All-Reduce liên tiếp với dung lượng dữ liệu siêu nhỏ ở mỗi layer, khiến hệ thống rơi vào trạng thái **Communication-bound**.
*   **NVLink** cung cấp băng thông cực lớn (>500 GB/s) và độ trễ cực thấp giúp tối ưu hóa pha Decode, trong khi **PCIe** quá chậm và sẽ gây nghẽn cổ chai nghiêm trọng nếu chạy Tensor Parallelism.
*   **NCCL Tuning**: Việc tinh chỉnh các biến môi trường như `NCCL_DEBUG`, `NCCL_BUFFSIZE`, và `CUDA_DEVICE_MAX_CONNECTIONS` là bắt buộc để gỡ lỗi và tối ưu hóa hiệu năng truyền thông liên-GPU trên thực tế production.
