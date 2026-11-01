"""
slope/compare.py — so hai trường mái dốc (FEM với FEM ở lưới khác, hoặc PINN với FEM).

Chạy được ở cả hai môi trường (chỉ cần numpy, scipy):

    python -m slope.compare results_slope/gate1_srf13.npz results_slope/gate1_srf13_fine.npz

File thứ nhất là THAM CHIẾU. File thứ hai được nội suy về các điểm của tham
chiếu. Ba chỉ số (chốt trước khi có PINN, xem README):

  M1  sai số L2 tương đối của chuyển vị trên các nút P2 của tham chiếu
  M2  sai số tương đối của |u_đỉnh|
  M3  IoU vùng dẻo: tập phần tử có dgam > DGAM_THR (ngưỡng TUYỆT ĐỐI, chung
      cho cả hai trường).

Vì sao ngưỡng tuyệt đối: dgam lớn nhất nằm ở góc chân dốc, là điểm kỳ dị,
nên nó KHÔNG hội tụ theo lưới (lưới 0.5 -> 0.25: 0.0092 -> 0.0183). Ngưỡng
theo tỉ lệ của max vì thế đổi theo lưới (5% max phủ 19.9% phần tử ở lưới thô
nhưng 3.5% ở lưới mịn). Ngưỡng tuyệt đối 5e-4 cho IoU giữa hai lưới 0.90.

Ngưỡng cổng 1 (chốt 26/09, trước khi có PINN), tham chiếu gate1_srf13_fine:
  M1 <= 0.05  (sàn FEM 0.0029)   M2 <= 0.05  (sàn 0.0004)   M3 >= 0.60 (sàn 0.90)
"""
import json, sys
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

DGAM_THR = 5e-4
GATE1 = dict(M1_L2_rel=("<=", 0.05), M2_crest_rel=("<=", 0.05), M3_iou=(">=", 0.60))


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def metrics(ref, oth):
    Xr, Ur = ref["coords"], ref["u"]
    lin = LinearNDInterpolator(oth["coords"], oth["u"])
    Uo = lin(Xr)
    bad = np.isnan(Uo).any(axis=1)                       # điểm sát biên nằm ngoài bao lồi
    if bad.any():
        Uo[bad] = NearestNDInterpolator(oth["coords"], oth["u"])(Xr[bad])
    m1 = np.linalg.norm(Uo - Ur) / np.linalg.norm(Ur)

    ur, uo = float(ref["u_crest_mag"]), float(oth["u_crest_mag"])
    m2 = abs(uo - ur) / ur

    thr = DGAM_THR
    dg_o = NearestNDInterpolator(oth["centroids"], oth["dgam"])(ref["centroids"])
    A, B = ref["dgam"] > thr, dg_o > thr
    m3 = (A & B).sum() / max((A | B).sum(), 1)
    return dict(M1_L2_rel=m1, M2_crest_rel=m2, M3_iou=m3,
                crest_ref_mm=ur * 1e3, crest_oth_mm=uo * 1e3,
                plastic_frac_ref=A.mean(), plastic_frac_oth=B.mean(),
                n_outside_hull=int(bad.sum()))


if __name__ == "__main__":
    ref, oth = load(sys.argv[1]), load(sys.argv[2])
    m = metrics(ref, oth)
    print(f"  tham chiếu: {sys.argv[1]}\n  so với    : {sys.argv[2]}")
    print(f"  M1 L2 tương đối của u       = {m['M1_L2_rel']:.4f}")
    print(f"  M2 lệch |u_đỉnh| tương đối  = {m['M2_crest_rel']:.4f}"
          f"   ({m['crest_ref_mm']:.3f} so với {m['crest_oth_mm']:.3f} mm)")
    print(f"  M3 IoU vùng dẻo             = {m['M3_iou']:.3f}"
          f"   (dẻo: {m['plastic_frac_ref']:.1%} so với {m['plastic_frac_oth']:.1%})")
    ok = True
    for key, (op, t) in GATE1.items():
        good = m[key] <= t if op == "<=" else m[key] >= t
        ok &= good
        print(f"  {key:14s} {op} {t:<5}  {'ĐẠT' if good else 'KHÔNG ĐẠT'}")
    print("  => cổng 1 " + ("ĐẠT" if ok else "KHÔNG ĐẠT"))
    if m["n_outside_hull"]:
        print(f"  ({m['n_outside_hull']} nút ngoài bao lồi, lấy láng giềng gần nhất)")
