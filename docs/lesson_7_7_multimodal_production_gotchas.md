---
sidebar_position: 11.88
sidebar_label: "Bài 7.7: Cấu hình Production & Gotchas trong Multimodal Serving"
---

# Bài 7.7: Cẩm nang Production và Tinh chỉnh hiệu năng LMMs

Triển khai phục vụ mô hình ngôn ngữ đa phương thức (Large Multimodal Models - LMMs) trong môi trường thực tế yêu cầu kỹ sư vận hành phải đối mặt với các vấn đề phức tạp hơn nhiều so với mô hình văn bản thuần túy. Sự bùng nổ token đột ngột từ hình ảnh/video và sự bất đối xứng về kích thước giữa bộ mã hóa thị giác (Vision Tower) và mô hình ngôn ngữ chính (LLM) là những nguyên nhân hàng đầu gây sụt giảm hiệu năng.

Bài học này sẽ cung cấp cẩm nang chi tiết cấu hình tham số CLI, phân tích lợi ích sống còn của Chunked Prefill đối với VLM và các thủ thuật tối ưu hóa phần cứng thực chiến.

---

## 1. Các tham số CLI quan trọng cho Multimodal Serving trong vLLM

Khi chạy API Server của vLLM phục vụ mô hình đa phương thức, bạn cần cấu hình và làm chủ các tham số CLI đặc thù sau để kiểm soát tài nguyên:

### A. `--limit-mm-per-prompt <limit_string>`
*   *Ý nghĩa*: Giới hạn số lượng vật thể đa phương tiện tối đa được phép xuất hiện trong một request.
*   *Ví dụ cấu hình*: `--limit-mm-per-prompt image=2,video=1`
*   *Tại sao quan trọng*: Một hình ảnh có thể tạo ra hàng trăm tokens, một video ngắn có thể tạo ra hàng ngàn tokens. Nếu không khống chế chủ động, người dùng có thể gửi các prompt chứa hàng chục tấm ảnh hoặc video dài, lập tức gây tràn bộ nhớ VRAM và làm sập tiến trình GPU của toàn hệ thống.

### B. `--mm-processor-cache-gb <float>`
*   *Ý nghĩa*: Cấu hình dung lượng bộ nhớ dùng cho cache đặc trưng thị giác (Vision Processor Cache). Giá trị mặc định thường là $4.0$ GB.
*   *Tại sao quan trọng*: Đặt giá trị này lớn hơn nếu hệ thống phục vụ các tác vụ hội thoại nhiều lượt (Multi-turn chat) kèm hình ảnh ổn định, giúp tái sử dụng ngay đặc trưng ảnh cũ mà không cần nạp lại.

### C. `--mm-shm-cache-max-object-size-mb <int>`
*   *Ý nghĩa*: Kích thước tối đa cho một vật thể đơn lẻ được phép lưu trữ trong Shared Memory cache (mặc định thường là $256$ MB).
*   *Tại sao quan trọng*: Nếu bạn phục vụ các mô hình phân tích video độ phân giải cao hoặc âm thanh dài, bạn cần tăng giá trị này lên để tránh các vật thể đặc trưng lớn bị từ chối cache và rơi về chế độ tính toán lại.

### D. `--trust-remote-code`
*   *Ý nghĩa*: Cho phép vLLM chạy mã nguồn tải trực tiếp từ repository của mô hình trên HuggingFace.
*   *Tại sao quan trọng*: Rất nhiều dòng mô hình LMM mới (như InternVL, Florence-2) sử dụng các định nghĩa lớp tùy chỉnh chưa được hợp nhất vào thư viện `transformers` chính thống. Nếu không bật cờ này, mô hình sẽ không thể load. Tuy nhiên, việc bật cờ này trong môi trường sản xuất có rủi ro bảo mật (chạy mã độc hại từ repo chưa kiểm định). Kỹ sư cần audit mã nguồn mô hình hoặc khóa mã hash commit của model repo.

### E. `--max-model-len <int>`
*   *Ý nghĩa*: Độ dài ngữ cảnh tối đa của mô hình mà server chấp nhận.
*   *Tại sao quan trọng*: Khác với văn bản ngắn, mỗi bức ảnh/video đưa vào sẽ tiêu tốn hàng trăm đến hàng ngàn tokens. Nếu người dùng gửi 3 tấm ảnh kèm câu hỏi, dung lượng prompt có thể vượt quá $2000$ tokens ngay lập tức. Nếu `--max-model-len` cấu hình quá thấp (ví dụ mặc định $2048$), hệ thống sẽ từ chối xử lý hoặc báo lỗi do không đủ ngữ cảnh để sinh câu trả lời. Bạn cần tính toán độ dài này hợp lý để bao phủ cả visual tokens và độ dài text mong muốn.

### F. `--gpu-memory-utilization <float>`
*   *Ý nghĩa*: Tỷ lệ bộ nhớ VRAM của GPU mà vLLM được phép sử dụng (mặc định $0.90$).
*   *Tại sao quan trọng*: Khi chạy LLM thuần văn bản, hầu hết VRAM còn lại được cấp phát cho KV Cache. Tuy nhiên, ở các mô hình VLM, Vision Tower và Projector chiếm dung lượng VRAM cố định đáng kể, đồng thời pha Prefill ảnh yêu cầu lượng bộ nhớ kích hoạt (activation memory) rất lớn. Nếu đặt tỷ lệ này quá sát ($0.95$), GPU rất dễ bị OOM trong quá trình prefill batch ảnh lớn. Đối với VLM, kỹ sư thường hạ tỷ lệ này xuống $0.80$ hoặc $0.85$ để tạo ra một khoảng đệm an toàn (headroom) cho Vision Tower hoạt động ổn định.

---

## 2. Chunked Prefill - Cứu cánh cho độ trễ của VLM

Một trong những vấn đề nghiêm trọng nhất của VLM serving là **sự gián đoạn cục bộ về độ trễ (Latency Spike)** khi nạp ảnh.

### Vấn đề khi tắt Chunked Prefill:
Khi một request chứa ảnh đi vào hệ thống, nó mang theo $576$ hoặc $1000$ visual tokens. Nếu không bật Chunked Prefill, vLLM bắt buộc phải xử lý toàn bộ $1000$ tokens này trong **đúng một bước forward duy nhất** của pha prefill.
*   GPU sẽ bị nghẽn tính toán GEMM lớn trong thời gian dài (vài trăm mili-giây).
*   Toàn bộ các request decode của người dùng khác đang chạy trong batch sẽ bị đóng băng (stall) để đợi pha prefill ảnh này kết thúc. Người dùng sẽ cảm nhận rõ rệt hiện tượng văn bản đang sinh ra mượt mà bỗng dưng bị khựng lại (lag).

### Giải pháp: Bật `--enable-chunked-prefill`
Bằng cách kích hoạt Chunked Prefill, vLLM sẽ tự động bẻ nhỏ chuỗi visual tokens khổng lồ của ảnh thành các mảnh nhỏ (chunks) cố định (ví dụ chunk size $256$ tokens):

```
Chuỗi nhúng ảnh 1000 tokens:
[ Chunk 0: 256 toks ] ➔ [ Chunk 1: 256 toks ] ➔ [ Chunk 2: 256 toks ] ➔ [ Chunk 3: 232 toks ]
```

Mỗi bước lặp (iteration), hệ thống chỉ prefill 1 chunk của ảnh đan xen với việc sinh token decode cho các request khác. Cơ chế lập lịch đan xen này giúp loại bỏ hoàn toàn hiện tượng khựng lag, ổn định chỉ số **Inter-Token Latency (ITL)** cho toàn bộ hệ thống phục vụ.

---

## 3. Cạm bẫy song song hóa Vision Tower (Tensor Parallelism vs ViT)

Đây là một cạm bẫy thiết kế hệ thống cực kỳ sâu sắc mà các kỹ sư phục vụ đa GPU thường mắc phải.

### Bản chất bất đối xứng về kích thước mô hình:
*   Mô hình ngôn ngữ chính (LLM) thường có dung lượng rất lớn (ví dụ 70B parameters), cần chạy trên 4 hoặc 8 GPU vật lý với cấu hình Tensor Parallelism (`--tensor-parallel-size 8`).
*   Bộ mã hóa thị giác Vision Tower (ViT) đi kèm lại có dung lượng rất nhỏ (thường chỉ khoảng 300M đến 1B parameters).

### Hậu quả của việc chia nhỏ mô hình quá bé:
Khi bạn cấu hình `TP=8` cho toàn hệ thống, theo mặc định, vLLM cũng sẽ cắt ma trận trọng số của Vision Tower ra làm 8 phần để chạy song song trên 8 GPU.
*   Do Vision Tower quá nhỏ, ma trận tính toán trên mỗi GPU cực kỳ tí hon.
*   Tuy nhiên, ở mỗi lớp của Vision Tower, các GPU vẫn phải thực hiện giao tiếp NCCL All-Reduce để đồng bộ dữ liệu.
*   Thời gian NCCL giao tiếp chéo giữa 8 GPU lúc này lớn hơn gấp nhiều lần thời gian GPU tính toán ma trận ViT cục bộ. Việc chạy Vision Tower trên 8 GPU kết quả là **chạy chậm hơn** rất nhiều so với việc chạy nó tập trung trên đúng 1 GPU duy nhất!

**Mẹo tối ưu hóa trong Production:**
*   Hạn chế lạm dụng cấu hình TP quá lớn cho các mô hình đa phương thức nếu không có NVLink tốc độ cực cao.
*   Ưu tiên sử dụng cấu hình song song dữ liệu (**Data Parallelism - DP Replicas**) kết hợp với TP nhỏ (ví dụ TP=2 hoặc TP=4) để giữ cho các mảnh ma trận Vision Tower đủ lớn trên từng GPU, giảm tối đa overhead truyền thông NCCL vô ích.

---

## 4. Liên hệ với Toy Engine: Sự phức tạp của Client đa phương tiện

Hãy đối chiếu mô hình phục vụ này với tệp kịch bản kiểm thử [client.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/toy_engine/client.py) trong Toy Serving Engine của chúng ta:

*   **Toy Client (Văn bản phẳng đơn giản)**: Trong kịch bản kiểm thử của Toy Engine, client chỉ việc gửi các chuỗi string prompt thuần túy có dung lượng vài byte qua HTTP POST. Quy trình gửi và nhận cực kỳ nhanh và không có nguy cơ lỗi truyền tải file.
*   **Production VLM Client (File nhị phân dung lượng lớn)**: Trong thực tế serving VLM, client phải truyền tải các tệp ảnh nhị phân dung lượng lớn (thường được mã hóa dạng chuỗi Base64 hoặc truyền qua URL liên kết hình ảnh/video). API Server của vLLM lúc này phải duy trì một luồng xử lý bất đồng bộ phức tạp: tải ảnh từ internet về, giải mã ảnh nhị phân, xử lý các lỗi ảnh hỏng, ảnh không đúng định dạng, và quản lý timeout tải file để tránh nghẽn luồng xử lý chính. Điều này biến API Gateway của VLM thành một hệ thống chịu tải I/O phức tạp hơn rất nhiều so với mô phỏng văn bản phẳng đơn giản.

---

## 💡 Tổng kết bài học

*   Các cờ CLI như `--limit-mm-per-prompt` và `--mm-processor-cache-gb` là bắt buộc để quản lý tài nguyên VRAM GPU và tối ưu hóa bộ đệm đặc trưng ảnh trên production.
*   Bật `--enable-chunked-prefill` là **cứu cánh** giúp bẻ nhỏ visual tokens của ảnh, loại bỏ hoàn toàn hiện tượng khựng lag độ trễ (latency spikes) cho các request đang hoạt động.
*   Tránh cấu hình **Tensor Parallelism (TP)** quá lớn đối với các bộ mã hóa thị giác Vision Tower nhỏ để ngăn ngừa overhead giao tiếp NCCL hủy hoại tốc độ chạy.
