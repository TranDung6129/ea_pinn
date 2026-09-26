"""
measure_fem_error.py — đo sai số rời rạc hoá của oracle FEM.
So E giữa hai độ mịn lưới. Con số này là căn cứ đặt cfg.FSR_TAU.

Khác bản cũ: truyền nx thẳng vào solve_reaction_diffusion thay vì gán
cfg.FEM_NX giữa chừng. Gán vào cfg là sửa trạng thái toàn cục, dễ rò sang
các lần chạy sau trong cùng một phiên.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config as cfg
from src.fem_oracle import solve_reaction_diffusion

NX_LOW, NX_HIGH = 15, 30

CASES = [
    (4.0,  4.0, 0.007, "gần biên, beta cao"),
    (5.0,  2.5, 0.01,  "gần biên, beta giữa"),
    (6.0,  1.5, 0.01,  "gần biên, beta thấp (knee)"),
    (3.0,  3.0, 0.15,  "gần biên, D cao"),
    (10.0, 1.0, 0.01,  "sập"),
    (14.0, 0.5, 0.005, "sập mạnh"),
    (2.0,  4.0, 0.1,   "ổn định"),
]


def main():
    print(f"Sai số rời rạc hoá FEM: NX={NX_LOW} so với NX={NX_HIGH}\n")
    diffs = []
    for a, b, D, label in CASES:
        e_lo = solve_reaction_diffusion(a, b, D, nx=NX_LOW)["E"]
        e_hi = solve_reaction_diffusion(a, b, D, nx=NX_HIGH)["E"]
        d = abs(e_lo - e_hi)
        diffs.append(d)
        print(f"  {label:<28} α={a:4.1f} β={b:4.1f} D={D:6.4f}  "
              f"E={e_lo:+.4f} / {e_hi:+.4f}  lệch {d:.4f}")

    mx = float(np.max(diffs))
    print(f"\n  trung bình {np.mean(diffs):.4f}   lớn nhất {mx:.4f}")
    print(f"  τ đề xuất ≈ {mx*1.5:.4f}   (cfg.FSR_TAU hiện tại = {cfg.FSR_TAU})")
    if mx * 1.5 > cfg.FSR_TAU * 1.5:
        print("  ⚠ τ trong config nhỏ hơn sai số đo được — cân nhắc nâng lên.")


if __name__ == "__main__":
    main()