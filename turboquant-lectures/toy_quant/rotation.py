"""
rotation.py — Trụ cột 1 của TurboQuant: phép xoay ngẫu nhiên (random rotation).

Hiện thực Randomized Hadamard Transform (RHT):  R = (1/sqrt(d)) * H * D
  - H: ma trận Hadamard (±1), nhân trong O(d log d) qua Fast Walsh–Hadamard Transform.
  - D: ma trận chéo các dấu ngẫu nhiên ±1 (random sign flip), sinh từ seed.

Tính chất quan trọng (xem Bài 2):
  - Bảo toàn chuẩn và tích vô hướng:  ||R x|| = ||x||,  <R x, R y> = <x, y>.
  - Đảo ngược được:  R^T (R x) = x   (vì R trực giao).
  - Không cần lưu ma trận, chỉ cần seed  ->  data-oblivious.
"""
from __future__ import annotations
import numpy as np


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh–Hadamard Transform (chưa chuẩn hóa). Yêu cầu len(a) là lũy thừa của 2.

    Thỏa mãn  fwht(fwht(a)) = len(a) * a  (H đối xứng, H^2 = d I).
    """
    x = a.astype(np.float64).copy()
    d = x.shape[0]
    assert d & (d - 1) == 0, "Độ dài phải là lũy thừa của 2"
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            block = x[i:i + 2 * h]
            u = block[:h].copy()
            v = block[h:].copy()
            block[:h] = u + v
            block[h:] = u - v
        h *= 2
    return x


class RandomRotation:
    """Phép xoay ngẫu nhiên dựa trên Randomized Hadamard Transform.

    Encoder và decoder chỉ cần CHIA SẺ cùng `seed` và `dim` — không truyền ma trận.
    """

    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self.pad = _next_pow2(dim)          # đệm tới lũy thừa của 2 cho Hadamard
        rng = np.random.default_rng(seed)
        # Dấu ngẫu nhiên ±1 cho D (chỉ cần lưu seed -> tái tạo được)
        self.signs = rng.choice(np.array([-1.0, 1.0]), size=self.pad)
        self.scale = 1.0 / np.sqrt(self.pad)

    def _embed(self, x: np.ndarray) -> np.ndarray:
        if self.pad == self.dim:
            return x.astype(np.float64)
        out = np.zeros(self.pad, dtype=np.float64)
        out[: self.dim] = x
        return out

    def rotate(self, x: np.ndarray) -> np.ndarray:
        """y = R x = (1/sqrt(d)) * H * (D x)."""
        z = self._embed(x) * self.signs
        return self.scale * fwht(z)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """x = R^T y = D * (1/sqrt(d)) * H * y  (cắt về dim gốc)."""
        z = self.scale * fwht(y) * self.signs
        return z[: self.dim]


if __name__ == "__main__":
    # Kiểm chứng các tính chất vàng của phép xoay
    rng = np.random.default_rng(123)
    d = 128
    R = RandomRotation(d, seed=42)
    x = rng.standard_normal(d)
    y = rng.standard_normal(d)

    rx, ry = R.rotate(x), R.rotate(y)

    print("== Kiểm chứng RandomRotation ==")
    print(f"Bảo toàn chuẩn        : ||x||={np.linalg.norm(x):.4f}  ||Rx||={np.linalg.norm(rx):.4f}")
    print(f"Bảo toàn tích vô hướng : <x,y>={x@y:.4f}  <Rx,Ry>={rx@ry:.4f}")
    print(f"Đảo ngược (sai số)     : {np.linalg.norm(R.inverse(rx) - x):.2e}")

    # Diệt outlier: nhồi một outlier rồi xem độ lớn tọa độ lớn nhất giảm thế nào
    xo = np.full(d, 0.05)
    xo[7] = 14.7
    print(f"\nMax|tọa độ| trước xoay : {np.max(np.abs(xo)):.3f}")
    print(f"Max|tọa độ| sau xoay   : {np.max(np.abs(R.rotate(xo))):.3f}  (outlier bị trải đều)")
