---
sidebar_position: 4.5
sidebar_label: "Bài 2.1: Phân tách Kiến trúc Attention Backends"
---

# Bài 2.1: Phân tách Kiến trúc Attention: Memory Layout vs Compute Backends

Trong các bài học trước, chúng ta đã tìm hiểu sâu về **PagedAttention** dưới góc độ giải thuật quản lý bộ nhớ (Memory Layout) giúp giải quyết triệt để bài toán phân mảnh KV Cache.

Tuy nhiên, trong codebase thực tế của vLLM, **PagedAttention không đơn độc**. vLLM đã thực hiện một bước đột phá về mặt kiến trúc phần mềm: **Tách biệt hoàn toàn cấu trúc quản lý bộ nhớ (Memory Layout) ra khỏi nhân tính toán (Compute Kernels)**. Lớp thiết kế trừu tượng này cho phép vLLM linh hoạt lựa chọn hoặc tích hợp các thư viện tối ưu hóa Attention mạnh nhất hiện nay như **FlashAttention-2, FlashInfer, xFormers, hay PyTorch FlexAttention** tùy thuộc vào kiến trúc phần cứng GPU bên dưới.

Bài học này sẽ bóc tách chi tiết thiết kế giải giao tiếp này trong mã nguồn vLLM.

---

## 1. Bản chất Kiến trúc: Decoupling Memory Layout vs Compute Backend

Để hiểu tại sao cần sự phân tách này, hãy hình dung quy trình tính toán Attention trên GPU:

```
+--------------------------------------------------------+
| 1. Quản lý Bộ nhớ (Memory Layout): Paged KV Cache      |
|    - Cấp phát động các block 16/32 tokens.              |
|    - Quản lý ánh xạ qua Block Table.                   |
+--------------------------------------------------------+
                           |
                           v (Giao tiếp dữ liệu qua con trỏ bộ nhớ / Tensors)
                           |
+--------------------------------------------------------+
| 2. Nhân tính toán (Compute Backend / Attention Kernels)|
|    - Thực hiện phép nhân Q x K^T x V.                  |
|    - Tận dụng SRAM, chia nhỏ Tensor (Tiling).          |
|    - Các thư viện: FlashAttention-2, FlashInfer, v.v.   |
+--------------------------------------------------------+
```

1. **Memory Layout (Paged KV Cache)**: Quyết định *ở đâu* (VRAM address nào) chứa các vector Key và Value của token thứ $t$ thuộc request thứ $r$.
2. **Compute Backend (Attention Kernels)**: Quyết định *làm sao* để nhân các vector Query ($Q$) với Key ($K$) và Value ($V$) đã được phân trang đó một cách nhanh nhất trên các thanh ghi và lõi Tensor Cores của GPU.

Nếu viết gộp chung, mỗi lần muốn tối ưu phép toán Attention cho một phần cứng mới (ví dụ GPU AMD hay Intel Gaudi), lập trình viên sẽ phải viết lại cả bộ quản lý bộ nhớ. Bằng cách tách biệt, vLLM chỉ cần duy trì một cấu trúc Paged KV Cache chung, và truyền các con trỏ block này vào các nhân Compute Kernels chuyên biệt phù hợp với từng GPU.

---

## 2. Cơ chế Tự động Lựa chọn: `vllm/v1/attention/selector.py`

Trong codebase vLLM (đặc biệt là kiến trúc vLLM V1 mới nhất), việc quyết định sử dụng thư viện tính toán nào được giao cho bộ chọn **Attention Selector** nằm tại `vllm/v1/attention/selector.py`.

### 2.1. Cấu hình Lựa chọn (`AttentionSelectorConfig`)
Bộ chọn sẽ đóng gói các tham số của mô hình và phần cứng thành một cấu trúc dữ liệu cấu hình:

```python
class AttentionSelectorConfig(NamedTuple):
    head_size: int                  # Kích thước chiều ẩn của Attention Head (ví dụ: 128)
    dtype: torch.dtype              # Kiểu dữ liệu mô hình (FP16, BF16, FP8)
    kv_cache_dtype: CacheDType | None # Kiểu dữ liệu lưu trong KV Cache
    block_size: int | None          # Kích thước khối (16 hoặc 32)
    use_mla: bool = False           # Có dùng Multi-head Latent Attention (MLA của DeepSeek) hay không
    attn_type: str = AttentionType.DECODER
    # ... các tùy chọn nâng cao khác
```

### 2.2. Luồng thực thi lựa chọn
Hàm `get_attn_backend(...)` sẽ được gọi trong quá trình khởi tạo mô hình để lấy ra backend phù hợp:

```python
def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    # ...
) -> type[AttentionBackend]:
    # 1. Đọc cấu hình hiện tại của engine
    vllm_config = get_current_vllm_config()
    
    # 2. Đóng gói cấu hình selector
    attn_selector_config = AttentionSelectorConfig(...)
    
    # 3. Yêu cầu nền tảng (NVIDIA, AMD, CPU) trả về lớp xử lý tương ứng
    return _cached_get_attn_backend(
        backend=vllm_config.attention_config.backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
```

Nền tảng phần cứng (`current_platform` - ví dụ `CudaPlatform`) sẽ dựa trên Compute Capability của GPU để trả về Backend Class phù hợp:
```python
# vllm/platforms/cuda.py
def get_attn_backend_cls(backend, attn_selector_config, ...):
    # Nếu người dùng cấu hình thủ công qua biến môi trường VLLM_ATTENTION_BACKEND
    if backend == "FLASHINFER":
        return "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    elif backend == "FLASH_ATTN":
        return "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    
    # Tự động chọn (Auto-selection) dựa trên kiến trúc phần cứng:
    # - GPU Ampere trở lên (A100, H100, v.v.): ưu tiên FlashAttention hoặc FlashInfer
    # - GPU cũ hơn (T4): ưu tiên xFormers
```

---

## 3. Đi sâu vào các Compute Backends chính trong vLLM

vLLM hiện tại tích hợp các backend tính toán Attention chuyên biệt nằm trong thư mục `vllm/v1/attention/backends/`:

### 3.1. FlashAttention-2 Backend (`flash_attn.py`)
* **Kiến trúc phần cứng tối ưu**: NVIDIA GPU dòng Ampere (RTX 3090, A100) và Hopper (H100) hỗ trợ kiến trúc tính toán Tensor Cores thế hệ mới.
* **Nguyên lý hoạt động**: Tận dụng giải thuật FlashAttention-2 tối ưu hóa I/O bộ nhớ bằng cách chia ma trận tính toán Attention thành các khối nhỏ vừa vặn với dung lượng **SRAM** của SM, thực hiện tính toán song song hóa trên luồng block mà không ghi kết quả trung gian ra HBM (VRAM).
* **Ứng dụng**: Đây là lựa chọn mặc định và tối ưu nhất cho pha **Prefill** (xử lý Prompt song song) của các mô hình Transformer tiêu chuẩn.

### 3.2. FlashInfer Backend (`flashinfer.py`)
* **Kiến trúc phần cứng tối ưu**: Các dòng GPU NVIDIA cao cấp chạy suy luận LLM Serving hiệu năng cao.
* **Nguyên lý hoạt động**: FlashInfer là thư viện chuyên dụng tối ưu hóa toán tử Attention cho Serving được phát triển bởi Đại học Washington. Nó chứa các kernel được tinh chỉnh cực mạnh cho:
  * **Decode với Paged KV Cache**: Tối ưu hóa truy cập địa chỉ không liên tục của trang bộ nhớ.
  * **Grouped-Query Attention (GQA)**: Khấu hao và tái sử dụng bộ nhớ cache cực tốt khi số lượng đầu ghi Key-Value ít hơn đầu ghi Query.
  * **Speculative Decoding**: Phép toán verify song song các token nháp đòi hỏi cấu trúc KV Cache phức tạp.
* **Ứng dụng**: Được vLLM ưu tiên sử dụng để tăng tốc pha **Decode** khi hệ thống xử lý lượng Batch Size lớn trên GPU Hopper/Blackwell.

### 3.3. Triton Attention Backend (`triton_attn.py`)
* **Kiến trúc phần cứng tối ưu**: Các nền tảng GPU không phải NVIDIA (như AMD ROCm hoặc Intel GPU) hoặc chạy chế độ gộp toán tử tùy biến.
* **Nguyên lý hoạt động**: Toàn bộ thuật toán tính toán Attention được viết bằng ngôn ngữ **OpenAI Triton** (Python-based). Khi chạy, Triton Compiler sẽ dịch mã này thành mã máy trực tiếp cho GPU tương ứng.
* **Ứng dụng**: Mang lại tính khả chuyển (portability) cực cao cho vLLM, giúp chạy mượt mà trên phần cứng của nhiều hãng sản xuất khác nhau mà không cần cài đặt các thư viện CUDA độc quyền của NVIDIA.

### 3.4. FlexAttention Backend (`flex_attention.py`)
* **Kiến trúc phần cứng tối ưu**: PyTorch 2.5 trở lên.
* **Nguyên lý hoạt động**: Tận dụng giải pháp **FlexAttention** mới của PyTorch. Cho phép người dùng viết các hàm Attention Mask tùy biến bằng Python thông thường, sau đó `torch.compile` sẽ tự động sinh ra và tối ưu hóa các kernel Triton phù hợp trên GPU.
* **Ứng dụng**: Cực kỳ hữu ích khi chạy các biến thể mô hình mới có cơ chế Attention Mask phức tạp (như DocQA, mô hình có cửa sổ ngữ cảnh trượt phức tạp).

---

## 💡 Tóm tắt bài học

1. **Phân tách thiết kế**: vLLM tách biệt **Paged KV Cache Layout** (quản lý lưu trữ bộ nhớ) khỏi **Compute Backend** (nhân tính toán phép nhân ma trận).
2. **Lớp Selector**: Lớp `vllm/v1/attention/selector.py` tự động phát hiện loại GPU, kiểu dữ liệu để chọn backend tối ưu nhất lúc khởi chạy.
3. **Thư viện tối ưu**:
   * **FlashAttention-2** chuyên trị Prefill (xử lý prompt dài).
   * **FlashInfer** được tối ưu hóa đặc biệt cho Paged KV Cache trong pha Decode và GQA.
   * **Triton** mang lại khả năng chạy đa nền tảng phần cứng (AMD, Intel, NVIDIA).
   * **FlexAttention** đem đến sự linh hoạt tối đa cho các cơ chế mask tùy biến thông qua PyTorch compiler.
