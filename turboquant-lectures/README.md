# TurboQuant Internals — Phân tích thuật toán & Tích hợp vLLM

> Chuỗi bài giảng tiếng Việt phân tích chuyên sâu thuật toán **TurboQuant** của Google
> (*"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*,
> Zandieh–Daliri–Hadian–Mirrokni, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026)
> và cách **tích hợp nó vào vLLM** để nén KV Cache khi serving LLM.

Đây là một site [Docusaurus](https://docusaurus.io/) **độc lập**, được phát triển trong nhánh của
repo [`vllm-architecture-lectures`](https://github.com/tuandung222/vllm-architecture-lectures) như một
khóa học chị em với *vLLM Internals*.

---

## ⚡ TurboQuant trong một câu

Xoay ngẫu nhiên vector (random rotation) khiến mọi tọa độ tuân theo cùng phân phối Beta ≈ Gauss và gần
độc lập → chỉ cần một bộ lượng hóa vô hướng tối ưu cho từng tọa độ (gần cận Shannon, hằng số ~2.72) →
thêm 1-bit QJL trên residual để ước lượng tích vô hướng **không thiên lệch**. Kết quả: nén KV Cache
xuống **2.5–3.5 bit/kênh**, **data-oblivious**, chạy **online**.

## 📚 Nội dung khóa học

| Bài | Chủ đề |
| :-- | :-- |
| 0 | Nền tảng Vector Quantization & Rate-Distortion |
| 1 | Bài toán nén KV Cache & yêu cầu Data-Oblivious |
| 2 | **Trụ cột 1** — Random Rotation & Phân phối Beta |
| 3 | **Trụ cột 2** — MSE Scalar Quantizer (Lloyd–Max, ~2.72×) |
| 4 | **Trụ cột 3** — Inner Product & QJL Unbiased |
| 5 | **Tích hợp vào vLLM** KV Cache (PagedAttention, FP8, kernel) |
| 6 | Cận dưới lý thuyết & Tính tối ưu ($\frac{\sqrt3\pi}{2}\approx 2.72$) |
| 7 | Ứng dụng Nearest Neighbor Search & Vector DB |
| 8 | **Thực hành** — Tự code Toy TurboQuant bằng NumPy |

## 🚀 Chạy site cục bộ

```bash
cd turboquant-lectures
npm install
npm run start        # dev server tại http://localhost:3000
npm run build        # build tĩnh, phải 0 lỗi
```

## 💻 Chạy Toy Implementation

```bash
cd turboquant-lectures/toy_quant
pip install numpy
python3 rotation.py       # kiểm chứng phép xoay (bảo toàn chuẩn/IP, diệt outlier)
python3 quantizer.py      # méo Lloyd–Max vs lý thuyết 2.72; QJL unbiased
python3 turboquant.py     # đường ống end-to-end; demo khử bias QJL
python3 kv_cache_demo.py  # nén KV Cache giả, đo MSE & attention score
python3 benchmark.py      # méo vs cận dưới; Nearest Neighbor Search
```

## 📖 Tài liệu tham khảo

- Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026.
- [vLLM](https://github.com/vllm-project/vllm) — thư viện LLM serving.
- [vLLM Internals Lectures](https://github.com/tuandung222/vllm-architecture-lectures) — khóa học chị em.

---

*Biên soạn bởi tuandung222. Built with Docusaurus.*
