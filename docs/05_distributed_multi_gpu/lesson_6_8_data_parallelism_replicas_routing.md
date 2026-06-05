---
sidebar_position: 8.99
sidebar_label: "Bài 6.8: Data Parallelism & API Routing"
---

# Bài 6.8: Data Parallelism (DP) Replicas & Phân phối Request - Giải pháp tối ưu thông lượng

Trong môi trường phục vụ thực tế (Production Serving), mục tiêu tối thượng không chỉ là làm sao chạy một mô hình lớn nhanh nhất, mà là làm sao xử lý đồng thời hàng ngàn yêu cầu từ người dùng (High Concurrency) với chi phí tối ưu. Để giải quyết bài toán quy mô này, hệ thống bắt buộc phải sử dụng cơ chế song song dữ liệu **Data Parallelism (DP) Replicas**.

Bài học này sẽ mổ xẻ cơ chế điều phối đa bản sao phục vụ, so sánh các giải thuật định tuyến request ở API Gateway, và đi sâu vào giải pháp **Prefix-Aware Routing** để ngăn ngừa hiện tượng phân mảnh bộ nhớ đệm KV Cache.

---

## 1. Vai trò của Data Parallelism (DP) Replicas trong Serving

Khi phục vụ LLM, chúng ta cần phân biệt rõ ràng mục tiêu tối ưu hóa của hai nhóm kỹ thuật song song:

1.  **Model Parallelism (TP, PP, CP)**: Cắt mô hình để phân bổ lên các GPU khác nhau nhằm **giảm thời gian phản hồi (Latency)** của từng request đơn lẻ, hoặc để nhét vừa mô hình vượt quá dung lượng 1 GPU VRAM.
2.  **Data Parallelism (DP)**: Nhân bản mô hình thành nhiều bản sao (replicas) độc lập chạy song song nhằm **tăng lượng request xử lý đồng thời (Throughput / Concurrency)** của toàn hệ thống.

### Cấu hình lai (Hybrid Parallelism) phổ biến:
Trên một server vật lý lớn (ví dụ server HGX chứa 8x GPU H100), để phục vụ mô hình Llama-3-70B, một cấu hình tối ưu thường được áp dụng là:
*   **TP = 4**: Gom 4 GPU thành 1 nhóm song song tensor để nạp và chạy mô hình 70B nhanh nhất.
*   **DP = 2**: Nhân bản thành 2 nhóm độc lập chạy song song (Nhóm 1: GPU 0-3, Nhóm 2: GPU 4-7).

```
┌────────────────────────────────────────────────────────┐
│                      API Gateway                       │
└───────────────────────────┬────────────────────────────┘
                            │ (Định tuyến Request)
              ┌─────────────┴─────────────┐
              ▼                           ▼
  [ Replica 0 (TP=4) ]        [ Replica 1 (TP=4) ]
     GPU 0, 1, 2, 3              GPU 4, 5, 6, 7
```

Khi đó, hệ thống có thể xử lý đồng thời hai luồng batch requests hoàn toàn độc lập, nâng gấp đôi thông lượng xử lý của server.

---

## 2. Các giải thuật định tuyến request ở API Gateway

Khi hàng ngàn client gửi request đồng thời lên API Server, API Gateway (hoặc router của vLLM) phải quyết định chuyển tiếp request đó đến replica nào dựa trên các thuật toán load balancing:

### A. Round-Robin (Xoay vòng)
*   *Cơ chế*: Request 1 sang Replica 0, Request 2 sang Replica 1, rồi lặp lại.
*   *Nhược điểm*: Không quan tâm đến độ dài thực tế của request. Nếu Replica 0 nhận phải các request sinh văn bản siêu dài (Decode heavy) trong khi Replica 1 nhận request ngắn, Replica 0 sẽ bị quá tải nghiêm trọng trong khi Replica 1 nhàn rỗi.

### B. Least-Connections (Ít kết nối nhất)
*   *Cơ chế*: Định tuyến request đến replica có số lượng request đang hoạt động ít nhất.
*   *Ưu điểm*: Cân bằng tải tốt hơn Round-robin.
*   *Nhược điểm*: Vẫn chưa tối ưu cho LLM vì 1 request prefill dài (tốn rất nhiều tính toán và KV Cache) có thể nặng gấp 10 lần một request decode ngắn, dù cả hai đều tính là "1 kết nối".

### C. Queue-size & VRAM-aware Routing (Định tuyến theo VRAM & Hàng đợi)
*   *Cơ chế*: API Gateway truy vấn thời gian thực trạng thái hàng đợi (`waiting_queue_size`) và tỉ lệ block KV Cache trống của từng replica. Request sẽ được ưu tiên gửi đến replica có hàng đợi ngắn nhất và còn nhiều Physical Blocks tự do nhất trong VRAM.

---

## 3. Bài toán phân mảnh Prefix Cache & Prefix-Aware Routing

Khi bật cơ chế tự động tái sử dụng cache tiền tố (**Radix Attention / Prefix Caching**), bài toán định tuyến trở nên cực kỳ quan trọng. 

Giả sử người dùng gửi 2 câu hỏi liên tiếp sử dụng chung một tài liệu ngữ cảnh dài 10K tokens (ví dụ prompt RAG phân tích một báo cáo tài chính).

```
1. Request 1 (Prompt A) ──> Định tuyến sang [ Replica 0 ]
   Replica 0 tính toán Prefill và lưu 10K tokens KV Cache vào Radix Tree.

2. Request 2 (Prompt B) ──> Định tuyến ngẫu nhiên sang [ Replica 1 ]
   Replica 1 hoàn toàn KHÔNG có KV Cache của tài liệu này. 
   Hậu quả: Replica 1 phải chạy lại Prefill 10K tokens từ đầu (Prefill Penalty)!
```

Việc định tuyến ngẫu nhiên hoặc chỉ dựa vào Least-connections sẽ làm xé lẻ dữ liệu cache, giảm tỷ lệ trúng cache tiền tố (**Cache Hit Rate**) của toàn hệ thống xuống mức rất thấp, gây lãng phí hàng chục TFLOPs tính toán GPU vô ích.

### Giải pháp: Định tuyến nhận biết tiền tố (Prefix-Aware Routing)

Để tối ưu hóa, bộ định tuyến API Gateway phải thực hiện phân tích nội dung prompt trước khi gửi đi:

```
[ Request 2 (Có chung tài liệu ngữ cảnh) ]
               │
               ▼
   [ API Gateway (Prefix Router) ]
   - Tính toán hash của đoạn Prompt đầu tiên
   - Nhận diện Hash này đã được nạp trên Replica 0
               │
               ▼ (Định tuyến ưu tiên)
      [ Replica 0 (Cache Hit!) ]
      (Tái sử dụng ngay lập tức KV Cache có sẵn, sinh token ngay)
```

Bằng cách luôn chuyển tiếp các request có chung đoạn tiền tố (Common Prefix Hash) về **cùng một replica**, hệ thống tối đa hóa tỷ lệ trúng cache tiền tố trong VRAM, triệt tiêu hoàn toàn chi phí prefill lại, giúp giảm thời gian phản hồi (TTFT) xuống mức gần bằng 0 cho các request tiếp theo.

---

## 4. Liên hệ với Toy Engine: Quy mô hàng đợi đơn nhất

Hãy đối chiếu mô hình cân bằng tải đa replica này với [Toy Serving Engine](file:///Users/admin/TuanDung/vllm-architecture-lectures/toy_engine/) của chúng ta:

*   **Toy Engine (Đơn Replica - DP=1)**: Trong [app.py](file:///Users/admin/TuanDung/vllm-architecture-lectures/toy_engine/app.py), API Server của chúng ta chỉ quản lý một hàng đợi đầu vào duy nhất (`input_queue`) và đẩy trực tiếp vào một Engine Core duy nhất. Hàng đợi này hoạt động theo nguyên tắc FIFO đơn giản và Block Allocator quản lý VRAM tập trung.
*   **Production vLLM (Đa Replicas - DP > 1)**: API Server trong môi trường production không thể dùng một hàng đợi phẳng đơn giản. Nó phải duy trì bảng trạng thái của nhiều hàng đợi con độc lập tương ứng với từng replica. Đồng thời, API Server đóng vai trò là một **Prefix-Aware Router**, phải thực hiện băm prompt và định tuyến thông minh vào đúng queue của replica thích hợp để bảo vệ hiệu năng bộ đệm KV Cache trong VRAM của từng GPU.

---

## 💡 Tổng kết bài học

*   **Data Parallelism (DP) Replicas** nhân bản mô hình thành nhiều bản sao phục vụ độc lập để tối ưu hóa lượng request xử lý đồng thời (**Throughput**), phối hợp với TP/PP để giảm độ trễ (**Latency**).
*   Các thuật toán định tuyến cơ bản (Round-robin, Least-connections) bỏ qua dung lượng bộ nhớ KV Cache và tính chất ngữ cảnh của request.
*   **Prefix-Aware Routing** băm các đoạn tiền tố của prompt và định tuyến các request có chung ngữ cảnh vào cùng một replica, giúp bảo vệ tỷ lệ trúng đệm KV Cache (**Prefix Cache Hit Rate**) và loại bỏ chi phí prefill lặp lại.
