# 🤖 Agent Playbook: TurboQuant Internals — Kiến trúc, Rigor & Pedagogy

Playbook này dành cho các coding agent kế tiếp bảo trì/mở rộng kho bài giảng **TurboQuant Internals** (phân tích thuật toán TurboQuant của Google & cách tích hợp vào vLLM).

---

## 1. Triết lý sư phạm: Theory → Toy → Production

Mục tiêu là không để học viên "hand-wave" coi quantization là hộp đen. Mọi khái niệm lý thuyết phải ánh xạ tới:
1. **Toán học & lý thuyết thông tin** (rate-distortion, phân phối Beta, high-rate quantization, JL lemma).
2. **Toy simulator (`toy_quant/`)** — hiện thực NumPy đơn giản, chạy được, có log kiểm chứng.
3. **Codebase thực tế (vLLM)** — chỉ rõ nơi thuật toán cắm vào khi serving.

---

## 2. Bản đồ khái niệm (Theory ➔ Toy ➔ vLLM)

| Khái niệm | Toy implementation | Vị trí trong vLLM (v1) |
| :--- | :--- | :--- |
| **Random Rotation (RHT)** | `toy_quant/rotation.py` (`RandomRotation`, `fwht`) | (cần custom) trước `reshape_and_cache` & trong attention kernel |
| **MSE Scalar Quantizer** | `toy_quant/quantizer.py` (`ScalarQuantizer`) | tham số `kv_cache_dtype`, write path |
| **QJL unbiased inner product** | `toy_quant/quantizer.py` (`QJL`) | attention backend `vllm/v1/attention/backends/*` |
| **Đường ống hợp nhất** | `toy_quant/turboquant.py` (`TurboQuant`) | RFC tích hợp — xem `docs/lesson_5_vllm_integration.md` |
| **KV Cache paging (không đổi)** | `kv_cache_demo.py` | `vllm/v1/core/kv_cache_manager.py` |

---

## 3. Các con số & công thức PHẢI giữ chính xác

Đừng đơn giản hóa toán. Các hằng số sau là **cốt lõi** và đã được kiểm chứng bằng toy:

* **Rate-distortion Gauss**: $D(R) = \sigma^2 2^{-2R}$ — quy luật **6 dB/bit**.
* **Méo scalar quantizer tối ưu (high-rate)**: $D_{\text{SQ}}(b) = \frac{\sqrt3\pi}{2}\sigma^2 2^{-2b} \approx 2.72\,\sigma^2 2^{-2b}$.
* **Hằng số gap**: $\frac{\sqrt3\pi}{2} \approx 2.72$ — **không đổi theo bit-width & chiều**. Đây là "con số 2.7" của paper.
* **Phân phối tọa độ sau xoay**: $\tilde x_i^2 \sim \text{Beta}(\tfrac12, \tfrac{d-1}2)$, $\mathbb E[\tilde x_i^2]=1/d$, xấp xỉ $\mathcal N(0,1/d)$.
* **QJL estimator**: $\langle q,k\rangle \approx \sqrt{\pi/2}\,\frac{\lVert k\rVert}{m}\langle Sq, \operatorname{sign}(Sk)\rangle$ — **unbiased**.
* **Kết quả KV Cache (paper)**: trung tính ở **3.5 bit/kênh**, suy giảm biên ở **2.5 bit/kênh**.
* **NNS**: indexing ~0.0013s vs ~239s của PQ; recall cao hơn PQ.

> ⚠️ Lưu ý sư phạm về bias: bias inner product **chỉ lộ ra khi query tương quan với key**. Mọi demo bias PHẢI dùng query tương quan (xem `turboquant.py`), nếu không sẽ ra bias ≈ 0 và gây hiểu nhầm.

---

## 4. Ràng buộc Workspace (Gotchas)

### ⚠️ Gotcha A: Đây là site Docusaurus ĐỘC LẬP trong subfolder
Repo này (`turboquant-lectures/`) là một dự án Docusaurus **riêng**, nằm trong nhánh của `vllm-architecture-lectures` nhưng **không liên quan** tới site vLLM ở thư mục gốc. Mọi lệnh `npm` phải chạy **bên trong** `turboquant-lectures/`.

### ⚠️ Gotcha B: React (`src/`) vs Python (`toy_quant/`)
Docusaurus cần React component ở `src/`. Mã Python để ở `toy_quant/` (KHÔNG để trong `src/`, sẽ phá build). Import Python dùng đường dẫn phẳng (chạy từ trong `toy_quant/`).

### ⚠️ Gotcha C: KaTeX SRI hash
Giữ đúng SRI hash KaTeX `0.16.8` trong `docusaurus.config.ts`:
```
integrity: 'sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn'
```

### ⚠️ Gotcha D: FWHT yêu cầu chiều là lũy thừa của 2
`rotation.py` tự đệm (pad) tới lũy thừa của 2. Khi tích hợp vLLM, $d_{\text{head}}$ thường đã là $64/128$ (sẵn lũy thừa của 2) nên rất tiện.

---

## 5. Quy trình & Backlog cho agent tương lai

1. **Validate**: chạy `npm run build` trong `turboquant-lectures/` — phải 0 lỗi. Chạy lại các script `toy_quant/*.py` nếu sửa thuật toán; mọi con số trích trong `docs/*.md` PHẢI khớp output thực tế.
2. **Sidebar order**: dùng `sidebar_position` thập phân để chèn bài mới không phải đánh số lại.
3. **Feature backlog**:
   - [ ] Thêm baseline **Product Quantization** (sklearn KMeans) vào `benchmark.py` để so indexing time trực tiếp.
   - [ ] Thêm baseline **uniform INT4 không xoay** để minh họa tác hại outlier.
   - [ ] Viết một bài phụ về **trellis-coded quantization** (cách thực sự vượt hằng số 2.72).
   - [ ] Prototype Triton kernel hợp nhất rotation + low-bit dot + popcount QJL (Bài 5, mode B).
