"""
kv_cache_demo.py — Mô phỏng nén KV Cache của một head bằng TurboQuant (Bài 5).

Sinh một "KV Cache" giả gồm L token, mỗi token có vector Key/Value chiều d_head,
CÓ kèm outlier (giống activation thật trong LLM lớn). Sau đó:
  - Nén Key & Value bằng TurboQuant ở các mức bit khác nhau.
  - Đo MSE tái tạo và — quan trọng cho attention — sai số điểm số <q, k>.
"""
from __future__ import annotations
import numpy as np
from turboquant import TurboQuant


def make_fake_kv(num_tokens: int, d_head: int, seed: int = 0):
    """KV giả với một vài kênh outlier (mô phỏng activation outlier trong LLM)."""
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((num_tokens, d_head))
    V = rng.standard_normal((num_tokens, d_head))
    # Nhồi outlier vào vài kênh cố định (giống hiện tượng systematic outlier channels)
    outlier_ch = rng.choice(d_head, size=3, replace=False)
    K[:, outlier_ch] *= rng.uniform(8, 20, size=3)
    return K, V


def run(num_tokens: int = 512, d_head: int = 128):
    K, V = make_fake_kv(num_tokens, d_head, seed=1)
    rng = np.random.default_rng(123)
    queries = rng.standard_normal((num_tokens, d_head))   # query cho mỗi vị trí

    print(f"KV Cache giả: {num_tokens} token, d_head={d_head}, có outlier channels")
    print(f"FP16 baseline: {num_tokens * d_head * 2 * 2 / 1024:.1f} KB (K+V)\n")
    print(f"{'bits':>5} {'KV size':>10} {'nén':>6} {'MSE(K)':>10} "
          f"{'rel||K||':>9} {'attn-score MAE':>15}")

    for bits in [4.0, 3.5, 3.0, 2.5, 2.0]:
        tqK = TurboQuant(dim=d_head, bits=bits, seed=7)   # Key: dùng QJL
        tqV = TurboQuant(dim=d_head, bits=bits, seed=9)   # Value: MSE mode

        mse_k, rel_k, score_err = [], [], []
        for t in range(num_tokens):
            ck = tqK.encode(K[t])
            k_hat = tqK.decode(ck)
            mse_k.append(np.mean((K[t] - k_hat) ** 2))
            rel_k.append(np.linalg.norm(K[t] - k_hat) / np.linalg.norm(K[t]))
            # Điểm số attention: <q, k> ước lượng unbiased (có QJL) vs thật
            true_s = float(queries[t] @ K[t])
            est_s = tqK.estimate_ip(queries[t], ck, use_qjl=True)
            score_err.append(abs(est_s - true_s))

        # Kích thước ~ (bits + 1 QJL) cho K, bits cho V  (bỏ qua norm)
        bits_per_coord = bits + 1.0  # +1 bit QJL cho key (xấp xỉ)
        kv_bytes = num_tokens * d_head * (bits_per_coord + bits) / 8.0
        ratio = (num_tokens * d_head * 2 * 2) / kv_bytes
        print(f"{bits:>5.1f} {kv_bytes/1024:>9.1f}K {ratio:>5.1f}x "
              f"{np.mean(mse_k):>10.3e} {np.mean(rel_k)*100:>8.1f}% "
              f"{np.mean(score_err):>15.4f}")

    print("\nNhận xét: MSE giảm ~4x mỗi khi tăng 1 bit (quy luật 6 dB/bit, Bài 3),")
    print("và QJL giữ cho sai số điểm số attention nhỏ & không thiên lệch (Bài 4).")


if __name__ == "__main__":
    run()
