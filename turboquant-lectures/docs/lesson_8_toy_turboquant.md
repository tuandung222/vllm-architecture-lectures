---
sidebar_position: 9
sidebar_label: "Bài 8: Thực hành — Toy TurboQuant"
---

# Bài 8: Thực hành — Tự xây dựng Toy TurboQuant

Lý thuyết đã đủ. Giờ ta **tự code** toàn bộ TurboQuant bằng Python & NumPy để **kiểm chứng** ba tuyên bố cốt lõi đã học: (1) random rotation diệt outlier & bảo toàn hình học, (2) méo MSE gần cận tối ưu với hằng số ~2.72, (3) QJL cho ước lượng inner product **không thiên lệch**. Mã nguồn nằm trong thư mục [`toy_quant/`](https://github.com/tuandung222/vllm-architecture-lectures/tree/main/turboquant-lectures/toy_quant).

> 💡 **Triết lý kỹ thuật**: *Chứng minh thuật toán đúng ở Python trước, rồi mới tối ưu CUDA kernel.* Toy này chính là bước "chứng minh đúng" mà bạn nên làm trước khi đầu tư tích hợp vào vLLM (Bài 5).

---

## 1. Cấu trúc thư mục `toy_quant/`

| File | Ánh xạ lý thuyết | Vai trò |
| :--- | :--- | :--- |
| `rotation.py` | Bài 2 | Randomized Hadamard Transform (FWHT + random sign) |
| `quantizer.py` | Bài 3 & 4 | `ScalarQuantizer` (Lloyd–Max) + `QJL` (unbiased inner product) |
| `turboquant.py` | Bài 2–4 | Hợp nhất: `encode / decode / estimate_ip` |
| `kv_cache_demo.py` | Bài 5 | Mô phỏng nén KV Cache một head, đo MSE & attention score |
| `benchmark.py` | Bài 6 & 7 | Quét méo vs lý thuyết; demo Nearest Neighbor Search |

Cài đặt & chạy:

```bash
cd turboquant-lectures/toy_quant
pip install numpy
python3 rotation.py      # kiểm chứng phép xoay
python3 quantizer.py     # méo scalar quantizer + QJL unbiased
python3 turboquant.py    # đường ống end-to-end
python3 kv_cache_demo.py # nén KV cache giả
python3 benchmark.py     # méo vs lý thuyết + NNS
```

---

## 2. Trụ cột 1 — Random Rotation (`rotation.py`)

Điểm cốt lõi là **Fast Walsh–Hadamard Transform** $O(d\log d)$ và tính chất $\text{FWHT}(\text{FWHT}(a)) = d\cdot a$, cho phép đảo ngược dễ dàng:

```python
class RandomRotation:
    def rotate(self, x):                       # y = R x = (1/√d) H (D x)
        z = self._embed(x) * self.signs
        return self.scale * fwht(z)

    def inverse(self, y):                       # x = R^T y = D (1/√d) H y
        z = self.scale * fwht(y) * self.signs
        return z[: self.dim]
```

**Kết quả chạy thực tế** (`python3 rotation.py`):

```text
Bảo toàn chuẩn        : ||x||=10.5798  ||Rx||=10.5798
Bảo toàn tích vô hướng : <x,y>=-6.1051  <Rx,Ry>=-6.1051
Đảo ngược (sai số)     : 2.68e-15

Max|tọa độ| trước xoay : 14.700      ← có outlier
Max|tọa độ| sau xoay   : 1.419       ← outlier bị trải đều!
```

> ✅ Đúng như Bài 2: phép xoay **bảo toàn chuẩn & tích vô hướng tuyệt đối**, **đảo ngược chính xác** ($10^{-15}$), và **một outlier $14.7$ bị tán xạ** thành các tọa độ độ lớn $\le 1.42$.

---

## 3. Trụ cột 2 — MSE Scalar Quantizer (`quantizer.py`)

`ScalarQuantizer` học bảng Lloyd–Max **một lần** trên mẫu Gauss (data-oblivious), bằng cách lặp hai điều kiện centroid + nearest-neighbor (vectorize bằng `np.bincount`).

**Kết quả chạy thực tế** — méo bám sát công thức $2.72\cdot 2^{-2b}$:

```text
 bits   D thực nghiệm     2.72*2^-2b   tỷ lệ/Shannon
    2       1.174e-01      1.700e-01          1.88x
    3       3.449e-02      4.251e-02          2.21x
    4       9.499e-03      1.063e-02          2.43x
    5       2.505e-03      2.657e-03          2.56x
```

> ✅ Đúng như Bài 3 & 6: méo giảm ~$4\times$ mỗi bit (**6 dB/bit**), và tỷ lệ so với cận Shannon dao động quanh **hằng số ~2.4–2.7** (xấp xỉ giá trị tiệm cận $\frac{\sqrt3\pi}{2}\approx 2.72$ của lý thuyết high-rate).

---

## 4. Trụ cột 3 — QJL khử bias (`turboquant.py`)

Đây là thí nghiệm **thuyết phục nhất**. Ta nén một key $k$, rồi ước lượng $\langle q, k\rangle$ với các query $q$ **tương quan** với $k$ (đúng tình huống self-attention), **có** và **không có** QJL:

```python
for _ in range(3000):
    k = rng.standard_normal(d)
    q = k + 0.3 * rng.standard_normal(d)        # query tương quan với key
    ck = tq.encode(k)
    true_ip = float(q @ k)
    err_no.append(tq.estimate_ip(q, ck, use_qjl=False) - true_ip)
    err_yes.append(tq.estimate_ip(q, ck, use_qjl=True) - true_ip)
```

**Kết quả chạy thực tế** (`python3 turboquant.py`):

```text
== Inner product: bias khi CÓ vs KHÔNG có QJL (query tương quan key) ==
Không QJL : bias=-4.2007  (ước lượng THẤP hơn do shrinkage)
Có   QJL  : bias=-0.0025  (đã khử bias)
```

> ✅ **Chính xác như Bài 4!** Bộ lượng hóa MSE co giá trị về trọng tâm → ước lượng tích vô hướng **thấp một cách hệ thống** (bias $= -4.2$). Thêm **1-bit QJL trên residual** đưa bias về **gần 0** ($-0.0025$). Đây là bằng chứng số học cho tuyên bố "*unbiased inner product quantizer*" của paper.

> [!NOTE]
> Lưu ý quan trọng: bias **chỉ lộ ra khi query tương quan với key**. Nếu bạn thử với query ngẫu nhiên độc lập, $\mathbb E[\langle q, e\rangle]=0$ và sẽ **không** thấy bias — đó là lý do nhiều người bỏ sót vấn đề này. Attention thực tế có query/key tương quan, nên bias là **có thật và quan trọng**.

---

## 5. Mô phỏng nén KV Cache (`kv_cache_demo.py`)

Ta tạo KV Cache giả của một head ($512$ token, $d_{\text{head}}=128$) **có outlier channels** giống activation thật, rồi nén Key (inner-product mode) và Value (MSE mode):

```text
KV Cache giả: 512 token, d_head=128, có outlier channels
FP16 baseline: 256.0 KB (K+V)

 bits    KV size    nén     MSE(K)  rel||K||  attn-score MAE
  4.0      72.0K   3.6x  5.443e-02      9.1%          2.5280
  3.5      64.0K   4.0x  1.106e-01     13.0%          3.2801
  3.0      56.0K   4.6x  1.985e-01     17.4%          4.8911
  2.5      48.0K   5.3x  3.332e-01     22.4%          5.8550
  2.0      40.0K   6.4x  6.255e-01     31.0%          7.8735
```

> ✅ Nén KV Cache **3.6×–6.4×** so với FP16. Sai số tái tạo Key tăng đều đặn (6 dB/bit), khớp với vùng "trung tính 3.5 bit, suy giảm biên 2.5 bit" mà paper báo cáo — và là tỷ lệ nén bạn sẽ thu được khi thay FP16/FP8 KV bằng TurboQuant trong vLLM (Bài 5).

---

## 6. Benchmark & Nearest Neighbor Search (`benchmark.py`)

Cuối cùng, demo ứng dụng thứ hai (Bài 7) — tìm kiếm lân cận gần nhất trên CSDL $5000$ vector $256$ chiều:

```text
== (2) Nearest Neighbor Search (MIPS), dim=256, DB=5000, top-10 ==
Thời gian indexing 5000 vector : 2866.8 ms (chỉ rotate+quantize, KHÔNG K-means)
Recall@10 ở 4.0 bit         : 83.9%
```

> ✅ "Indexing" chỉ là **encode mỗi vector** — **không có bước K-means**. Đây chính là lợi thế "indexing tức thời" của TurboQuant so với Product Quantization (Bài 7). Recall@10 đạt **~84%** ở chỉ 4 bit/tọa độ (nén $8\times$ so với FP32), nhờ ước lượng inner product unbiased.

> [!TIP]
> Bài tập mở rộng cho bạn:
> 1. Viết một baseline **uniform INT4 KV (không xoay)** và so sánh MSE — bạn sẽ thấy outlier phá hỏng nó thế nào.
> 2. Hiện thực một baseline **Product Quantization** bằng `sklearn.cluster.KMeans` và đo thời gian indexing để so trực tiếp với TurboQuant.
> 3. Thay FWHT giả lập bằng thư viện `scipy.linalg.hadamard` và đo tốc độ ở $d=4096$.

---

## 7. Tổng kết toàn khóa

Qua 9 bài, ta đã đi trọn vẹn từ lý thuyết tới hiện thực:

* **Bài 0–1**: Bài toán — nén vector chiều cao bảo toàn cả MSE lẫn inner product, **online & data-oblivious**, cho KV Cache.
* **Bài 2–4**: Ba trụ cột — **random rotation** (Beta/Gauss, diệt outlier), **MSE scalar quantizer** (Lloyd–Max, ~2.72× cận tối ưu), **QJL** (unbiased inner product).
* **Bài 5**: Tích hợp thực tế vào **vLLM** — hai điểm cắm (write/read), so sánh FP8 KV Cache, custom kernel.
* **Bài 6**: Cận dưới thông tin & hằng số tối ưu $\frac{\sqrt3\pi}{2}\approx 2.72$.
* **Bài 7**: Ứng dụng **NNS/Vector DB** — indexing tức thời, recall cao.
* **Bài 8**: **Tự code & kiểm chứng** mọi tuyên bố bằng NumPy.

> 🎓 TurboQuant là một ví dụ đẹp về việc **một kết quả lý thuyết thông tin chặt chẽ** (rate-distortion + random rotation) trực tiếp tạo ra **giá trị kỹ thuật khổng lồ** — nén KV Cache cho LLM serving và tăng tốc vector database — chỉ với những phép toán đơn giản, rẻ tiền, không cần dữ liệu. Đó chính là vẻ đẹp của việc hiểu thuật toán tới tận gốc rễ thay vì coi nó là hộp đen.

---

**Tài liệu tham khảo chính**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (2025), ICLR 2026.
