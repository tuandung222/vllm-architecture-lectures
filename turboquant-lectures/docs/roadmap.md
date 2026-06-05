---
sidebar_position: 0
sidebar_label: "🗺️ Roadmap & Syllabus"
---

# TurboQuant Internals: Phân tích thuật toán & Tích hợp vLLM

Chào mừng bạn đến với chuỗi bài giảng **TurboQuant Internals**. Đây là tài liệu phân tích chuyên sâu (bằng tiếng Việt) về thuật toán **TurboQuant** — công trình *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"* của nhóm tác giả **Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni** (Google Research & Google DeepMind, ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

Chuỗi bài giảng được thiết kế cho **AI Serving Engineer, ML Research Engineer và Systems Engineer** muốn hiểu thấu đáo:
1. **Bản chất toán học** của TurboQuant (rate-distortion, random rotation, scalar quantization, QJL).
2. **Cách tích hợp thực tế** thuật toán này vào **vLLM** để nén KV Cache khi serving LLM.

> 🎯 **Mục tiêu tối thượng**: Không "hand-wave" coi quantization là hộp đen. Mọi khái niệm lý thuyết phải ánh xạ trực tiếp tới (1) công thức toán & cận thông tin, (2) toy implementation trong `toy_quant/`, và (3) vị trí code thực tế trong mã nguồn vLLM.

---

## 🧩 TurboQuant trong một đoạn

> **Bài toán**: Cho một vector chiều cao $x \in \mathbb{R}^d$ (ví dụ một vector Key/Value trong KV Cache), hãy nén nó xuống còn $b$ bit/tọa độ sao cho **méo (distortion)** nhỏ nhất — đo bằng cả **sai số bình phương trung bình (MSE)** lẫn **sai số tích vô hướng (inner product)** — mà **không cần dữ liệu hiệu chỉnh (calibration-free)** và chạy được **online**.
>
> **Lời giải TurboQuant** gồm 3 trụ cột:
> 1. **Random Rotation**: Xoay ngẫu nhiên $x$ → mọi tọa độ tuân theo cùng một phân phối Beta đã biết, gần như độc lập.
> 2. **MSE Scalar Quantizer**: Áp một bộ lượng hóa vô hướng tối ưu (Lloyd-Max) cho từng tọa độ độc lập.
> 3. **QJL Residual**: Thêm 1-bit Quantized Johnson–Lindenstrauss trên phần dư để khử thiên lệch (bias) khi ước lượng tích vô hướng.
>
> **Kết quả**: Méo chỉ cách **cận dưới lý thuyết thông tin** một hằng số nhỏ (~2.7) đồng đều ở mọi bit-width và mọi chiều. Với KV Cache: trung tính chất lượng ở **3.5 bit/kênh**, suy giảm không đáng kể ở **2.5 bit/kênh**.

---

## 🗺️ Lộ trình Chuỗi Bài Giảng (Roadmap)

| Bài học | Chủ đề | Nội dung cốt lõi | Loại |
| :--- | :--- | :--- | :--- |
| **[Bài 0](./lesson_0_vector_quantization_background.md)** | **Nền tảng Vector Quantization** | Lý thuyết mã hóa nguồn Shannon, scalar vs vector quantization, hàm méo rate-distortion $D(R)=\sigma^2 2^{-2R}$, hai loại distortion (MSE vs inner product). | Background |
| **[Bài 1](./lesson_1_kv_cache_problem.md)** | **Bài toán KV Cache & Data-Oblivious** | Vì sao KV Cache nghẽn VRAM, phân biệt PTQ tĩnh (calibration) vs lượng hóa online, vì sao cần thuật toán data-oblivious. | Core Theory |
| **[Bài 2](./lesson_2_random_rotation_beta.md)** | **Trụ cột 1 — Random Rotation** | Toán học xoay ngẫu nhiên, phân phối Beta của tọa độ trên mặt cầu, tính gần-độc-lập ở chiều cao, randomized Hadamard transform $O(d\log d)$. | Core Theory |
| **[Bài 3](./lesson_3_mse_scalar_quantizer.md)** | **Trụ cột 2 — MSE Scalar Quantizer** | Lloyd-Max quantizer, lý thuyết high-rate quantization, công thức méo $D(b)\propto 2^{-2b}$, vì sao "chia để trị" theo tọa độ là gần tối ưu. | Core Theory |
| **[Bài 4](./lesson_4_inner_product_qjl.md)** | **Trụ cột 3 — Inner Product & QJL** | Vì sao MSE-quantizer thiên lệch khi ước lượng tích vô hướng, Quantized JL transform, ước lượng inner product không thiên lệch (unbiased). | Core Theory |
| **[Bài 5](./lesson_5_vllm_integration.md)** | **Tích hợp vào vLLM** | Ánh xạ thuật toán vào PagedAttention, attention backend, online quant trong decode loop, so sánh FP8 KV Cache, custom kernel. | Integration |
| **[Bài 6](./lesson_6_lower_bound_optimality.md)** | **Cận dưới & Tính tối ưu** | Cận dưới information-theoretic, vì sao hằng số gap $\approx \frac{\sqrt 3 \pi}{2}\approx 2.72$, ý nghĩa "near-optimal at all bit-widths". | Theory Deep Dive |
| **[Bài 7](./lesson_7_nearest_neighbor_search.md)** | **Ứng dụng NNS & Vector DB** | Thay thế Product Quantization trong ANN search, recall cao hơn, indexing gần như tức thời (0.0013s vs 239s). | Application |
| **[Bài 8](./lesson_8_toy_turboquant.md)** | **Thực hành — Toy TurboQuant** | Tự code đường ống TurboQuant bằng NumPy, mô phỏng nén KV Cache, đo MSE & sai số inner product để kiểm chứng lý thuyết. | Practice |

---

## 📂 Cấu trúc Repository

```bash
turboquant-lectures/
├── README.md                              # Giới thiệu tổng quan
├── docs/                                  # 9 bài giảng lý thuyết & phân tích
│   ├── lesson_0_vector_quantization_background.md
│   ├── lesson_1_kv_cache_problem.md
│   ├── lesson_2_random_rotation_beta.md
│   ├── lesson_3_mse_scalar_quantizer.md
│   ├── lesson_4_inner_product_qjl.md
│   ├── lesson_5_vllm_integration.md
│   ├── lesson_6_lower_bound_optimality.md
│   ├── lesson_7_nearest_neighbor_search.md
│   └── lesson_8_toy_turboquant.md
└── toy_quant/                             # Toy implementation (Bài 8)
    ├── rotation.py                        # Randomized Hadamard transform (random rotation)
    ├── quantizer.py                       # MSE scalar quantizer + QJL residual
    ├── turboquant.py                      # Encoder/Decoder hoàn chỉnh (quantize ↔ dequantize)
    ├── kv_cache_demo.py                   # Mô phỏng nén KV Cache & đo distortion
    └── benchmark.py                       # Quét MSE vs bit-width, demo recall NNS
```

---

## 🛠️ Yêu cầu chuẩn bị (Prerequisites)

1. **Đại số tuyến tính**: tích vô hướng, chuẩn Euclid, ma trận trực giao (orthogonal), phép xoay.
2. **Xác suất & thống kê**: kỳ vọng, phương sai, phân phối Gauss, định lý giới hạn trung tâm (CLT).
3. **Lý thuyết thông tin cơ bản** (không bắt buộc): entropy, rate-distortion — sẽ được nhắc lại ở Bài 0.
4. **Python & NumPy**: để chạy `toy_quant/` ở Bài 8.
5. **(Khuyến khích)** Đã đọc qua chuỗi [vLLM Internals](https://github.com/tuandung222/vllm-architecture-lectures) để nắm KV Cache & PagedAttention — phần Bài 5 sẽ tích hợp trực tiếp.

---

## 🚀 Bắt đầu như thế nào?

Bạn nên đọc tuần tự từ **[Bài 0](./lesson_0_vector_quantization_background.md)** để có nền tảng rate-distortion, sau đó đi qua 3 trụ cột (Bài 2–4). Nếu bạn là **serving engineer** và muốn vào thẳng phần ứng dụng, có thể nhảy tới **[Bài 5: Tích hợp vào vLLM](./lesson_5_vllm_integration.md)** rồi quay lại đọc lý thuyết sau.
