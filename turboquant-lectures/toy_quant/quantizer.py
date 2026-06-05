"""
quantizer.py — Trụ cột 2 & 3 của TurboQuant.

  - ScalarQuantizer : bộ lượng hóa vô hướng tối ưu MSE (Lloyd–Max) cho nguồn Gauss (Bài 3).
  - QJL             : Quantized Johnson–Lindenstrauss, ước lượng tích vô hướng KHÔNG
                      thiên lệch (unbiased) chỉ với 1 bit/chiều chiếu (Bài 4).
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# Trụ cột 2: Lloyd–Max scalar quantizer (tối ưu MSE cho phân phối đã biết)
# ----------------------------------------------------------------------------
class ScalarQuantizer:
    """Bộ lượng hóa vô hướng b-bit tối ưu MSE cho nguồn chuẩn tắc N(0,1).

    Bảng mức (levels) và biên (boundaries) được học MỘT LẦN offline bằng thuật
    toán Lloyd trên mẫu Gauss -> data-oblivious (không phụ thuộc dữ liệu người dùng).
    """

    def __init__(self, bits: float, n_samples: int = 500_000, iters: int = 80, seed: int = 0):
        self.bits = bits
        self.K = max(2, int(round(2 ** bits)))     # số mức tái tạo = 2^b
        rng = np.random.default_rng(seed)
        samples = rng.standard_normal(n_samples)
        self.levels = self._lloyd(samples, self.K, iters)
        # Biên quyết định = trung điểm giữa các mức liền kề (điều kiện nearest-neighbor)
        self.boundaries = (self.levels[:-1] + self.levels[1:]) / 2.0

    @staticmethod
    def _lloyd(samples: np.ndarray, K: int, iters: int) -> np.ndarray:
        # Khởi tạo mức bằng các phân vị (quantile) cho phân bố đều theo xác suất
        qs = (np.arange(K) + 0.5) / K
        levels = np.quantile(samples, qs)
        for _ in range(iters):
            bounds = (levels[:-1] + levels[1:]) / 2.0
            idx = np.searchsorted(bounds, samples)          # gán mẫu vào ô gần nhất
            # Cập nhật centroid (vectorize bằng bincount cho nhanh & chính xác ở đuôi)
            sums = np.bincount(idx, weights=samples, minlength=K)
            counts = np.bincount(idx, minlength=K).astype(np.float64)
            new = np.where(counts > 0, sums / np.maximum(counts, 1), levels)
            if np.allclose(new, levels, atol=1e-7):
                break
            levels = new
        return levels

    def quantize(self, u: np.ndarray) -> np.ndarray:
        """u (đã chuẩn hóa ~ N(0,1)) -> chỉ số mức (int)."""
        return np.searchsorted(self.boundaries, u).astype(np.int32)

    def dequantize(self, idx: np.ndarray) -> np.ndarray:
        return self.levels[idx]

    def distortion_per_coord(self) -> float:
        """MSE thực nghiệm của quantizer trên N(0,1)."""
        rng = np.random.default_rng(7)
        u = rng.standard_normal(1_000_000)
        return float(np.mean((u - self.dequantize(self.quantize(u))) ** 2))


# ----------------------------------------------------------------------------
# Trụ cột 3: QJL — ước lượng tích vô hướng unbiased, 1 bit/chiều chiếu
# ----------------------------------------------------------------------------
class QJL:
    """Quantized Johnson–Lindenstrauss.

    Lưu cho mỗi vector k:  sign(S k)  (m bit) và  ||k||.
    Ước lượng KHÔNG thiên lệch:
        <q, k> ~= sqrt(pi/2) * (||k|| / m) * < S q , sign(S k) >.
    """

    def __init__(self, dim: int, m: int, seed: int = 0):
        self.dim = dim
        self.m = m
        rng = np.random.default_rng(seed)
        self.S = rng.standard_normal((m, dim))     # ma trận chiếu Gauss (chia sẻ qua seed)
        self.c = np.sqrt(np.pi / 2.0)

    def encode(self, k: np.ndarray):
        """Trả về (sign bits, norm) — phần lưu trữ nén của k."""
        return np.sign(self.S @ k).astype(np.int8), float(np.linalg.norm(k))

    def estimate(self, q: np.ndarray, signs: np.ndarray, knorm: float) -> float:
        """Ước lượng <q, k> chỉ từ q (full precision), dấu của k, và ||k||."""
        return self.c * knorm / self.m * float((self.S @ q) @ signs)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1) Kiểm chứng méo scalar quantizer khớp công thức 2.72 * 2^(-2b)
    print("== ScalarQuantizer: méo vs lý thuyết high-rate ==")
    print(f"{'bits':>5} {'D thực nghiệm':>15} {'2.72*2^-2b':>14} {'tỷ lệ/Shannon':>14}")
    for b in [2, 3, 4, 5, 6]:
        sq = ScalarQuantizer(bits=b)
        D = sq.distortion_per_coord()
        theo = (np.sqrt(3) * np.pi / 2) * 2 ** (-2 * b)   # sigma^2 = 1
        shannon = 2 ** (-2 * b)
        print(f"{b:>5} {D:>15.3e} {theo:>14.3e} {D/shannon:>13.2f}x")

    # 2) Kiểm chứng QJL là unbiased cho inner product
    print("\n== QJL: tính unbiased của ước lượng inner product ==")
    d, m = 256, 1024
    qjl = QJL(d, m, seed=1)
    q = rng.standard_normal(d)
    k = rng.standard_normal(d)
    true_ip = float(q @ k)
    signs, knorm = qjl.encode(k)
    # Trung bình nhiều ma trận chiếu độc lập -> hội tụ về giá trị thật (unbiased)
    ests = []
    for s in range(500):
        qjl_s = QJL(d, m, seed=100 + s)
        sg, kn = qjl_s.encode(k)
        ests.append(qjl_s.estimate(q, sg, kn))
    print(f"<q,k> thật            : {true_ip:.3f}")
    print(f"QJL trung bình (500x) : {np.mean(ests):.3f}  (bias ~ {np.mean(ests)-true_ip:+.3f})")
