"""
turboquant.py — Hợp nhất 3 trụ cột thành đường ống TurboQuant hoàn chỉnh.

Sơ đồ (Bài 2–4):
    x ─► [Random Rotation] ─► tách norm ─► [MSE Scalar Quantizer] ─► residual ─► [1-bit QJL]

Hai chế độ:
    - decode(code)        : tái tạo vector x̂ (MSE mode).
    - estimate_ip(q, code): ước lượng <q, x> KHÔNG thiên lệch (inner-product mode, dùng QJL).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rotation import RandomRotation
from quantizer import ScalarQuantizer, QJL


@dataclass
class TurboCode:
    """Biểu diễn nén của một vector."""
    idx: np.ndarray         # chỉ số mức scalar quantizer (b bit/tọa độ)
    norm: float             # ||x|| (lượng hóa riêng; ở đây giữ float cho toy)
    qjl_signs: np.ndarray   # dấu QJL trên residual (1 bit/chiều chiếu)
    qjl_rnorm: float        # ||residual|| để chuẩn hóa ước lượng QJL

    def num_bits(self, sq_bits: float) -> float:
        """Ước lượng số bit (bỏ qua chi phí hằng số của norm)."""
        return self.idx.size * sq_bits + self.qjl_signs.size  # +1 bit/chiều QJL


class TurboQuant:
    def __init__(self, dim: int, bits: float = 3.0, qjl_dim: int | None = None, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.rot = RandomRotation(dim, seed=seed)
        self.pad = self.rot.pad
        self.sq = ScalarQuantizer(bits=bits, seed=seed)
        self.sqrt_pad = np.sqrt(self.pad)
        m = qjl_dim if qjl_dim is not None else self.pad      # mặc định m = pad (≈ +1 bit/tọa độ)
        self.qjl = QJL(self.pad, m, seed=seed + 1)

    # ----------------------------- ENCODE -----------------------------
    def encode(self, x: np.ndarray) -> TurboCode:
        xt = self.rot.rotate(x)                  # xoay -> tọa độ ~ Beta ≈ Gauss
        n = float(np.linalg.norm(xt))            # tách độ lớn
        if n == 0:
            n = 1e-12
        w = self.sqrt_pad * (xt / n)             # chuẩn hóa về ~ N(0,1)
        idx = self.sq.quantize(w)                # lượng hóa vô hướng từng tọa độ

        xt_hat = n * (self.sq.dequantize(idx) / self.sqrt_pad)  # tái tạo trong không gian xoay
        resid = xt - xt_hat                       # residual cho QJL
        signs, rnorm = self.qjl.encode(resid)
        return TurboCode(idx=idx, norm=n, qjl_signs=signs, qjl_rnorm=rnorm)

    # ------------------------- DECODE (MSE mode) ----------------------
    def decode(self, code: TurboCode) -> np.ndarray:
        xt_hat = code.norm * (self.sq.dequantize(code.idx) / self.sqrt_pad)
        return self.rot.inverse(xt_hat)

    # ------------------- ESTIMATE INNER PRODUCT (unbiased) ------------
    def estimate_ip(self, q: np.ndarray, code: TurboCode, use_qjl: bool = True) -> float:
        qt = self.rot.rotate(q)                            # xoay query bằng CÙNG R
        xt_hat = code.norm * (self.sq.dequantize(code.idx) / self.sqrt_pad)
        ip = float(qt @ xt_hat)                            # phần MSE (có bias)
        if use_qjl:
            ip += self.qjl.estimate(qt, code.qjl_signs, code.qjl_rnorm)  # khử bias
        return ip


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, bits = 128, 3.0
    tq = TurboQuant(dim=d, bits=bits, seed=42)

    x = rng.standard_normal(d)
    code = tq.encode(x)
    x_hat = tq.decode(code)

    mse = float(np.mean((x - x_hat) ** 2))
    rel = float(np.linalg.norm(x - x_hat) / np.linalg.norm(x))
    print("== TurboQuant end-to-end (1 vector) ==")
    print(f"dim={d}, bits={bits}  ->  ~{code.num_bits(bits)/d:.2f} bit/tọa độ (gồm QJL)")
    print(f"MSE tái tạo            : {mse:.4e}")
    print(f"Sai số tương đối ||·||  : {rel*100:.2f}%")

    # So sánh ước lượng inner product CÓ vs KHÔNG có QJL.
    # Lưu ý: bias chỉ lộ ra khi query TƯƠNG QUAN với key (ví dụ self-attention,
    # q ≈ k). Với query ngẫu nhiên độc lập, E[<q,e>]=0 nên không thấy bias.
    # Ta đo trên nhiều cặp (key, query=key+nhiễu) — đúng tình huống attention.
    print("\n== Inner product: bias khi CÓ vs KHÔNG có QJL (query tương quan key) ==")
    err_no, err_yes = [], []
    for _ in range(3000):
        k = rng.standard_normal(d)
        q = k + 0.3 * rng.standard_normal(d)       # query tương quan với key
        ck = tq.encode(k)
        true_ip = float(q @ k)
        err_no.append(tq.estimate_ip(q, ck, use_qjl=False) - true_ip)
        err_yes.append(tq.estimate_ip(q, ck, use_qjl=True) - true_ip)
    print(f"Không QJL : bias={np.mean(err_no):+.4f}  (ước lượng THẤP hơn do shrinkage)")
    print(f"Có   QJL  : bias={np.mean(err_yes):+.4f}  (đã khử bias)")
