---
sidebar_position: 9
sidebar_label: "Bài 7: Kỹ thuật Tối ưu hóa Nâng cao"
---

# Bài 7: Kỹ thuật Tối ưu hóa Nâng cao cho AI Serving

Bên cạnh các cơ chế phân trang bộ nhớ (PagedAttention) và lập lịch thông minh (Continuous Batching), vLLM còn tích hợp nhiều kỹ thuật tối ưu hóa tiên tiến nhất trong nghiên cứu hệ thống học sâu để tăng tốc độ suy luận và tiết kiệm tài nguyên. Trong bài học này, chúng ta sẽ khảo sát ba kỹ thuật nâng cao: **Speculative Decoding**, **Multi-LoRA Serving**, và các định dạng lượng hóa bộ nhớ (**Quantization**).

---

## 1. Speculative Decoding (Giải mã suy đoán)

### Ý tưởng cốt lõi:
Nhắc lại từ Bài 1: Pha Decode bị nghẽn bộ nhớ (Memory-bound) do phải nạp trọng số mô hình lớn (ví dụ 70B) từ HBM chỉ để xử lý duy nhất 1 token ở mỗi bước.
**Speculative Decoding** giải quyết vấn đề này bằng cách kết hợp hai mô hình:
1. **Draft Model (Mô hình nháp)**: Có kích thước rất nhỏ (ví dụ Llama 68M hoặc 1B) nhưng chạy cực kỳ nhanh.
2. **Target Model (Mô hình mục tiêu)**: Mô hình lớn chất lượng cao mà chúng ta muốn phục vụ (ví dụ Llama 70B).

```
Quy trình thực hiện:
Bước 1: Draft Model tự hồi quy sinh nhanh K tokens nháp (ví dụ K = 5).
Bước 2: Target Model nhận cả Prompt + K tokens nháp này làm đầu vào, chạy DUY NHẤT 1 bước forward song song (Prefill-style).
Bước 3: Target Model tính toán phân phối xác suất và quyết định chấp nhận (Accept) hay từ chối (Reject) từng token nháp:
        - Chấp nhận các token khớp với phân phối của nó.
        - Thay thế token bị từ chối đầu tiên bằng một token chính xác.
        - Hủy bỏ toàn bộ các token nháp phía sau token bị từ chối đó.
```

Nhờ cơ chế này, thay vì phải chạy Target Model $K$ lần độc lập, chúng ta chỉ cần chạy $1$ lần duy nhất. Nếu chấp nhận được trung bình $3$ trên $5$ tokens nháp, tốc độ sinh token tổng thể của hệ thống sẽ tăng từ **1.5x đến 2.5x**.

### Quản lý KV Cache trong Speculative Decoding:
Trong vLLM, việc tích hợp Speculative Decoding đòi hỏi cơ chế quản lý khối bộ nhớ KV Cache rất phức tạp:
* Trình quản lý bộ nhớ phải cấp phát trước các khối trống dự phòng cho các token nháp của Draft Model.
* Khi Target Model từ chối một số token nháp (ví dụ chấp nhận 2, từ chối 3 từ cuối): vLLM phải lập tức thực hiện cắt bỏ (Prune) các slot tương ứng của các token nháp bị từ chối khỏi KV Cache của Target Model và giải phóng các khối vật lý trống đó trở lại Block Pool.

---

## 2. Multi-LoRA Serving (Phục vụ đa Adapter)

Trong môi trường thực tế, người dùng muốn tinh chỉnh mô hình nền (Base Model) cho nhiều nhiệm vụ khác nhau bằng các Adapter LoRA (nhỏ, dung lượng khoảng vài chục MB). 
* **Thử thách**: Nếu chạy mỗi bản LoRA trên một GPU độc lập, chúng ta sẽ cạn kiệt phần cứng. Nếu nạp và hủy LoRA liên tục khi có request tương ứng, độ trễ nạp trọng số từ CPU sang GPU qua PCIe sẽ phá hỏng tốc độ serving.
* **Giải pháp của vLLM (Multi-LoRA Serving)**:
  * vLLM giữ duy nhất 1 bản Base Model trong VRAM GPU.
  * Hệ thống nạp sẵn nhiều adapter LoRA khác nhau vào các vùng nhớ đệm (LoRA Cache) trên VRAM GPU.
  * Bộ lập lịch xếp các request sử dụng các LoRA khác nhau vào chung một Batch để thực thi Continuous Batching.

```
Base Model: [   Q, K, V Matrices   ] (Dùng chung cho toàn bộ batch)
                |          |
Request 1:  [ + LoRA A ]   |
Request 2:                 [ + LoRA B ]
```

### Thuật toán thực thi Multi-LoRA:
Khi chạy forward một batch gồm các request sử dụng các LoRA khác nhau:
* vLLM sử dụng các kernel GPU chuyên dụng như **Punica** hoặc **SGMV** (Segmented Gather Matrix-Vector multiplication).
* Các kernel này cho phép thực hiện phép nhân ma trận trọng số chung của Base Model cho cả lô, sau đó tự động gom nhóm các token tương ứng với từng LoRA để nhân với các adapter weights khác nhau một cách hiệu quả, loại bỏ hoàn toàn hiện tượng tuần tự hóa tính toán.

---

## 3. Quantization (Lượng hóa Bộ nhớ)

Lượng hóa là kỹ thuật giảm số bit biểu diễn cho trọng số mô hình hoặc giá trị KV Cache để tiết kiệm VRAM và tăng tốc tính toán. vLLM tích hợp sâu với các phương pháp lượng hóa hàng đầu:

### Lượng hóa KV Cache (FP8 / INT8 KV Cache):
* **Nguyên lý**: Thay vì lưu KV Cache ở định dạng mặc định FP16 (2 Bytes/token), vLLM chuyển đổi dữ liệu sang định dạng FP8 (1 Byte/token) hoặc INT8 trước khi lưu vào khối vật lý.
* **Kết quả**: Dung lượng bộ nhớ KV Cache giảm đi **một nửa** (đối với FP8) hoặc nhiều hơn, cho phép nhân đôi kích thước tối đa của Batch size hoặc chiều dài ngữ cảnh trên cùng một GPU.

### Lượng hóa Trọng số (Weight-Only Quantization - AWQ, GPTQ):
* **Nguyên lý**: Lượng hóa trọng số mô hình xuống INT4 hoặc INT8, nhưng giữ nguyên kích hoạt (Activation) ở dạng FP16 khi tính toán để đảm bảo chất lượng mô hình không bị suy giảm.
* **Tích hợp trong vLLM**: vLLM sử dụng các kernel tối ưu hóa cực cao như **Marlin** hoặc **AWQ GEMM Kernels** để thực hiện giải nén nhanh trọng số từ INT4/INT8 sang FP16 ngay trên SRAM của GPU trước khi nhân. Điều này giúp giảm đáng kể băng thông bộ nhớ đọc mô hình trong pha Decode.

---

## 💡 Tổng kết bài học
* **Speculative Decoding** tận dụng mô hình nhỏ sinh nháp và dùng mô hình lớn kiểm chứng song song, giảm số lần chạy mô hình lớn và đẩy nhanh tốc độ sinh token.
* **Multi-LoRA Serving** cho phép chạy đồng thời hàng chục mô hình tinh chỉnh khác nhau trên cùng một GPU bằng cách kết hợp Base Model và các custom kernel chuyên biệt (Punica/SGMV).
* **Quantization** (FP8 KV Cache, AWQ/GPTQ INT4 Weights) giảm dung lượng lưu trữ của KV Cache và trọng số mô hình trực tiếp, tăng mật độ batch và throughput phục vụ.

Trong bài học cuối cùng tiếp theo, chúng ta sẽ thiết kế cấu trúc và tự tay viết mã nguồn cho một **Toy Serving Engine** hoàn chỉnh để hiện thực hóa các kiến thức lý thuyết đã học từ đầu chuỗi bài giảng!
