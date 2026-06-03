# Bài 2: Đột phá của vLLM – Thuật toán PagedAttention & Cấp phát Khối

Trong bài 1, chúng ta đã hiểu lý do tại sao các hệ thống phục vụ LLM truyền thống bị giới hạn bởi vấn đề phân mảnh bộ nhớ KV Cache. Trong bài này, chúng ta sẽ phân tích giải pháp mang tính cách mạng của vLLM: **PagedAttention**. Đây là phát minh cốt lõi giúp vLLM tăng năng lực phục vụ gấp hàng chục lần bằng cách tối ưu hóa triệt để không gian bộ nhớ GPU.

---

## 1. Ý tưởng cốt lõi: Phân trang Bộ nhớ ảo (Virtual Memory Paging)

Trong các hệ điều hành máy tính (Windows, Linux, macOS), khi một ứng dụng yêu cầu cấp phát bộ nhớ RAM, hệ thống không tìm một vùng nhớ liên tục khổng lồ trong RAM vật lý. Thay vào đó, hệ điều hành sử dụng **Bộ nhớ ảo (Virtual Memory)**:
* Bộ nhớ của ứng dụng được chia thành các trang logic có kích thước cố định (**Pages**).
* RAM vật lý được chia thành các khung trang vật lý (**Page Frames**).
* Hệ điều hành quản lý một **Bảng trang (Page Table)** để ánh xạ các trang logic (có vẻ liên tục đối với ứng dụng) vào các khung trang vật lý nằm rải rác khắp nơi trong RAM.

```
Ứng dụng nhìn thấy bộ nhớ Logic liên tục:
[ Trang 0 ] -> [ Trang 1 ] -> [ Trang 2 ] -> [ Trang 3 ]

Bảng trang (Page Table):
Trang 0 -> Khung vật lý 45
Trang 1 -> Khung vật lý 12
Trang 2 -> Khung vật lý 98
Trang 3 -> Khung vật lý 03

RAM vật lý thực tế (không liên tục):
[ Khung 03: Trang 3 ] ... [ Khung 12: Trang 1 ] ... [ Khung 45: Trang 0 ] ... [ Khung 98: Trang 2 ]
```

vLLM áp dụng chính xác nguyên lý này lên bộ nhớ VRAM GPU để lưu trữ KV Cache. Kỹ thuật này gọi là **PagedAttention**.

---

## 2. Thiết kế Cấu trúc Khối: Logical Blocks vs Physical Blocks

Trong vLLM, KV Cache của mỗi Sequence không còn được lưu trữ dưới dạng một tensor liên tục khổng lồ nữa. Thay vào đó, nó được chia thành các **Khối bộ nhớ (Blocks)** có kích thước cố định đại diện cho một số lượng token nhất định (thường là `block_size = 16` hoặc `32` tokens).

### Khối Logic (Logical Blocks):
* Đối với mô hình và người dùng, KV Cache của một Sequence vẫn là một chuỗi các token liên tục.
* Ví dụ, một Sequence có 50 tokens. Với `block_size = 16`, Sequence này được chia thành 4 Khối Logic:
  * **Logical Block 0**: Chứa KV Cache của token từ 0 đến 15.
  * **Logical Block 1**: Chứa KV Cache của token từ 16 đến 31.
  * **Logical Block 2**: Chứa KV Cache của token từ 32 đến 47.
  * **Logical Block 3**: Chứa KV Cache của token từ 48 đến 49 (chưa đầy, còn trống 14 slot).

### Khối Vật lý (Physical Blocks):
* Trong không gian bộ nhớ GPU VRAM, vLLM định nghĩa sẵn một vùng nhớ trống khổng lồ gọi là **Block Pool**. Vùng nhớ này được cắt sẵn thành các khối vật lý có kích thước bằng đúng kích thước của Khối Logic.
* Các khối vật lý này được cấp phát động cho các khối logic khi mô hình sinh token mới. Các khối vật lý ứng với một Sequence có thể nằm **rải rác và không liên tục** trên VRAM.

### Bảng ánh xạ Khối (Block Mapping Table):
Bộ quản lý bộ nhớ của vLLM (`BlockSpaceManager` hoặc `KVCacheManager` trong v1) duy trì một bảng ánh xạ để biết khối logic của Sequence nào tương ứng với khối vật lý nào trên GPU.

```
Sequence A (50 tokens, Block Size = 16):
Logical Blocks:  [ Block 0 ]   [ Block 1 ]   [ Block 2 ]   [ Block 3 ]
                      |             |             |             |
Page Table:           v             v             v             v
Physical Blocks: [ Phys 104 ]  [ Phys 42 ]   [ Phys 12 ]   [ Phys 90 ]
```

---

## 3. Hoạt động của Thuật toán PagedAttention Kernel (CUDA/Triton)

Attention truyền thống yêu cầu các ma trận Key ($K$) và Value ($V$) phải nằm liên tục trên bộ nhớ để GPU có thể thực hiện phép nhân ma trận nhanh chóng:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Nếu $K$ và $V$ nằm rải rác ở các địa chỉ vật lý khác nhau (`Phys 104`, `Phys 42`,...), các phép toán Attention tiêu chuẩn của PyTorch sẽ không thể thực thi được. 

Để giải quyết vấn đề này, các tác giả vLLM đã viết một **Custom CUDA Kernel** (sau này được tối ưu bằng **Triton**) để tính Attention trực tiếp trên các khối bộ nhớ phân trang.

```
Luồng tính toán PagedAttention Kernel cho 1 đầu Attention (Head):
1. Đọc vector Query (Q) của token hiện tại từ SRAM.
2. Dựa vào Page Table của Request, xác định danh sách các địa chỉ Physical Blocks của K và V.
3. Với mỗi Physical Block:
   a. Nạp các vector K tương ứng (16 tokens) vào bộ nhớ dùng chung của GPU (Shared Memory).
   b. Thực hiện nhân Q với K của khối đó -> Tính điểm số chú ý (Attention Scores).
   c. Lưu trữ kết quả tích lũy.
4. Chạy Softmax trên toàn bộ điểm số chú ý đã tính.
5. Với mỗi Physical Block:
   a. Nạp các vector V tương ứng (16 tokens) vào Shared Memory.
   b. Nhân điểm số chú ý đã chuẩn hóa với V -> Tính vector Output.
6. Trả về kết quả Output cuối cùng.
```

> [!TIP]
> **Tại sao block_size lại là 16 hoặc 32?**
> * Nếu block_size quá nhỏ (ví dụ = 1): Việc truy vấn bảng Page Table xảy ra liên tục, làm tăng độ trễ và GPU không thể tận dụng tính chất đọc tuần tự của bộ nhớ (coalesced memory access).
> * Nếu block_size quá lớn (ví dụ = 256): Phân mảnh bộ nhớ tăng lên vì phần trống ở khối cuối cùng của mỗi sequence không được tận dụng.
> * Thực nghiệm cho thấy `16` và `32` là điểm cân bằng hoàn hảo giữa hiệu năng truy cập bộ nhớ và giảm thiểu lãng phí VRAM.

---

## 4. Chia sẻ Bộ nhớ (Memory Sharing) qua Copy-on-Write (CoW)

Một trong những ứng dụng mạnh mẽ nhất của PagedAttention là khả năng chia sẻ KV Cache giữa các Sequence có chung tiền tố (Prefix) mà không cần nhân đôi bộ nhớ vật lý.

### Ứng dụng 1: Parallel Sampling (Sinh nhiều kết quả từ cùng Prompt)
Giả sử người dùng yêu cầu sinh 3 câu trả lời khác nhau từ cùng một Prompt dài 1000 tokens (ví dụ: viết 3 bản thảo email từ yêu cầu mô tả).
* **Truyền thống**: Hệ thống phải copy Prompt làm 3 bản riêng biệt và tính toán/lưu trữ KV Cache độc lập cho cả 3. Tiêu tốn bộ nhớ tăng gấp 3 lần.
* **vLLM (PagedAttention)**: Do Prompt là chung, 3 sequence này sẽ dùng chung các khối vật lý chứa KV Cache của Prompt. Chúng chỉ tạo ra các khối vật lý mới khi bắt đầu sinh ra các token khác nhau ở phần output.

```
Prompt chung: [ Block 0 ] -> [ Block 1 ] -> [ Block 2 ] (Dùng chung bộ nhớ vật lý)
                                                |
                 +------------------------------+------------------------------+
                 | (Khác nhau)                                                 | (Khác nhau)
                 v                                                             v
Seq 1: [ Phys Block 3 (Output 1) ]                            Seq 2: [ Phys Block 4 (Output 2) ]
```

### Ứng dụng 2: Cơ chế Copy-on-Write (CoW)
Khi hai sequence dùng chung một khối vật lý, khối vật lý đó được đánh dấu là **Read-Only (Chỉ đọc)** và có bộ đếm tham chiếu (Reference Count) bằng 2.
Nếu một trong hai sequence cần ghi thêm token mới vào khối đó (khi khối chưa đầy):
1. vLLM phát hiện bộ đếm tham chiếu $> 1$.
2. Hệ thống sẽ **sao chép (copy)** khối vật lý đó sang một khối vật lý trống mới.
3. Cập nhật bảng trang của sequence vừa ghi để trỏ vào khối mới này.
4. Giảm bộ đếm tham chiếu của khối cũ đi 1.
5. Tiến hành ghi token mới vào khối vật lý mới (Write).

Cơ chế này tương tự như quản lý bộ nhớ của tiến trình con (`fork()`) trong Linux, giúp tiết kiệm dung lượng VRAM GPU một cách tối đa.

---

## 💡 Tổng kết bài học
* **PagedAttention** giải quyết triệt để phân mảnh bộ nhớ bằng cách chia KV Cache thành các **khối vật lý kích thước cố định** (16/32 tokens) và ánh xạ chúng thông qua một bảng trang (Page Table).
* vLLM tự hiện thực một **custom Triton/CUDA kernel** để tính Attention trực tiếp trên bộ nhớ phân mảnh, tránh overhead copy dữ liệu liên tục.
* Cơ chế **Copy-on-Write** cho phép chia sẻ KV Cache cực kỳ hiệu quả trong các tác vụ Parallel Sampling, Beam Search, và lưu trữ tiền tố hệ thống (System Prompt Caching).

Trong bài học tiếp theo, chúng ta sẽ tìm hiểu cơ chế xếp lịch cấp cao của vLLM dựa trên kiến trúc phân trang này: **Continuous Batching & Preemption**.
