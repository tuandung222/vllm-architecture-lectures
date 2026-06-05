"""
benchmark.py — Kiểm chứng định lượng hai tuyên bố chính của TurboQuant.

  (1) Méo MSE bám sát công thức near-optimal:  D(b) ≈ (sqrt(3)*pi/2) * 2^(-2b)  (Bài 3, 6).
  (2) Demo Nearest Neighbor Search: TurboQuant cho recall cao với "indexing" tức thời (Bài 7).
"""
from __future__ import annotations
import time
import numpy as np
from turboquant import TurboQuant


def benchmark_distortion(dim: int = 256, n_vectors: int = 400):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_vectors, dim))
    sigma2 = 1.0  # phương sai mỗi tọa độ của nguồn chuẩn tắc

    print("== (1) Méo MSE vs cận dưới lý thuyết (near-optimal ~2.72x) ==")
    print(f"{'bits':>5} {'D đo được':>13} {'2.72·2^-2b':>13} {'D/Shannon':>11}")
    for bits in [2.0, 3.0, 4.0, 5.0]:
        tq = TurboQuant(dim=dim, bits=bits, seed=3)
        errs = []
        for x in X:
            x_hat = tq.decode(tq.encode(x))
            errs.append(np.mean((x - x_hat) ** 2))
        D = float(np.mean(errs))
        theo = (np.sqrt(3) * np.pi / 2) * sigma2 * 2 ** (-2 * bits)
        shannon = sigma2 * 2 ** (-2 * bits)
        print(f"{bits:>5.1f} {D:>13.3e} {theo:>13.3e} {D/shannon:>10.2f}x")


def benchmark_nns(dim: int = 256, n_db: int = 5000, n_query: int = 200, topk: int = 10, bits: float = 4.0):
    rng = np.random.default_rng(1)
    DB = rng.standard_normal((n_db, dim))
    Q = rng.standard_normal((n_query, dim))

    # Ground-truth Top-k theo inner product (full precision)
    true_ip = Q @ DB.T
    true_top = np.argsort(-true_ip, axis=1)[:, :topk]

    print(f"\n== (2) Nearest Neighbor Search (MIPS), dim={dim}, DB={n_db}, top-{topk} ==")
    tq = TurboQuant(dim=dim, bits=bits, seed=5)

    # ---- Indexing: chỉ encode mỗi vector (KHÔNG train/K-means) ----
    t0 = time.perf_counter()
    codes = [tq.encode(x) for x in DB]
    t_index = time.perf_counter() - t0

    # ---- Gom mã thành ma trận để truy vấn vectorized (chỉ là tối ưu tính toán) ----
    Xhat = np.stack([c.norm * (tq.sq.dequantize(c.idx) / tq.sqrt_pad) for c in codes])   # (n_db, pad)
    Signs = np.stack([c.qjl_signs.astype(np.float64) for c in codes])                    # (n_db, m)
    Rnorm = np.array([c.qjl_rnorm for c in codes])                                       # (n_db,)

    recalls = []
    for i in range(n_query):
        qt = tq.rot.rotate(Q[i])                                  # xoay query MỘT lần
        ip_mse = Xhat @ qt                                        # phần MSE cho mọi DB
        ip_qjl = tq.qjl.c * Rnorm / tq.qjl.m * (Signs @ (tq.qjl.S @ qt))   # phần QJL khử bias
        est = ip_mse + ip_qjl
        pred_top = np.argpartition(-est, topk)[:topk]
        recalls.append(len(set(pred_top) & set(true_top[i])) / topk)

    print(f"Thời gian indexing {n_db} vector : {t_index*1000:.1f} ms (chỉ rotate+quantize, KHÔNG K-means)")
    print(f"Recall@{topk} ở {bits} bit         : {np.mean(recalls)*100:.1f}%")
    print("So sánh: Product Quantization cần chạy K-means (chậm hơn nhiều bậc) để index.")


if __name__ == "__main__":
    benchmark_distortion()
    benchmark_nns()
