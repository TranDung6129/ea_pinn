"""
measure_dslope.py — không gọi oracle, chạy vài giây. Hai câu hỏi:

  1. Dấu của ∂E/∂D: so độ nghiêng dα*/dD của biên thật (lưới GT 25³) với biên PINN.
     Lưới GT lấy trung bình qua 25 lát nên không bị nhiễu rời rạc hoá như sai phân.

  2. Biên PINN có phải chỉ là biên thật bị dịch ngang không.
     Δα = α*_pinn − α*_thật tại từng (β, D). Nếu trong một lát D, Δα gần như hằng
     (tản mát nhỏ so với trung vị) thì sai số là dịch mức, không phải sai hình dạng.
     Dòng cuối cho biết một hiệu chỉnh tuyến tính theo (β, D) gỡ được bao nhiêu.
     Phần này dùng ground truth nên chỉ để chẩn đoán; hiệu chỉnh thật phải khớp
     trên chính 200 lời gọi oracle của phương pháp.

  python measure_dslope.py --tag cap --variant ls1_bs0_kn0
  python measure_dslope.py --tag cap --variant ls1_bs1_kn0
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import config as cfg
from src.ground_truth import generate_ground_truth
from src.pinn_model import build_model, normalise_params
from src.loss_functions import failure_functional

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="cap")
ap.add_argument("--seed", type=int, default=cfg.BASE_SEED)
ap.add_argument("--variant", default="ls1_bs0_kn0")
ap.add_argument("--n-alpha", type=int, default=97, help="độ mịn trục α khi dựng biên PINN")
args = ap.parse_args()
dev = cfg.DEVICE


def load_model():
    name = f"fmd_pinn_seed{args.seed}_{args.variant}" + (f"_{args.tag}" if args.tag else "")
    path = os.path.join(cfg.CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        cands = [f for f in os.listdir(cfg.CKPT_DIR)
                 if f.startswith(f"{name}_call") and f.endswith(".pt")]
        if not cands:
            sys.exit(f"không thấy checkpoint cho {name}")
        path = os.path.join(cfg.CKPT_DIR,
                            max(cands, key=lambda f: int(f.rsplit("call", 1)[1][:-3])))
    m = build_model(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    print(f"  checkpoint: {os.path.basename(path)}")
    return m


def E_grid(model, al, be, Dg):
    A, B, D = np.meshgrid(al, be, Dg, indexing="ij")
    flat = np.column_stack([A.ravel(), B.ravel(), D.ravel()])
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), 256):
            q = flat[i:i + 256]
            p = normalise_params(*[torch.tensor(q[:, k], dtype=torch.float32, device=dev)
                                   for k in range(3)])
            out.append(failure_functional(model, p, device=dev).cpu().numpy())
    return np.concatenate(out).reshape(A.shape)


def crossings(E, al):
    """α* tại mỗi (β_j, D_k): lần đổi dấu đầu tiên theo α, nội suy tuyến tính; NaN nếu không có."""
    a = np.full(E.shape[1:], np.nan)
    for j in range(E.shape[1]):
        for k in range(E.shape[2]):
            c = E[:, j, k]
            s = np.where(np.sign(c[:-1]) * np.sign(c[1:]) < 0)[0]
            if len(s):
                i = s[0]
                t = c[i] / (c[i] - c[i + 1])
                a[j, k] = al[i] + t * (al[i + 1] - al[i])
    return a


def slopes(a_star, Dg):
    """dα*/dD theo từng hàng β, khớp tuyến tính trên các lát D có biên."""
    out = np.full(a_star.shape[0], np.nan)
    for j in range(a_star.shape[0]):
        ok = np.isfinite(a_star[j])
        if ok.sum() >= 5:
            out[j] = np.polyfit(Dg[ok], a_star[j, ok], 1)[0]
    return out


if __name__ == "__main__":
    model = load_model()
    gt = generate_ground_truth()
    al, be, Dg = gt["alpha_grid"], gt["beta_grid"], gt["D_grid"]
    span_a = al.max() - al.min()

    a_gt = crossings(gt["E_fem"], al)
    al_f = np.linspace(al.min(), al.max(), args.n_alpha)
    a_pn = crossings(E_grid(model, al_f, be, Dg), al_f)

    # ── 1. dấu của D ──────────────────────────────────────────────────────────
    s_gt, s_pn = slopes(a_gt, Dg), slopes(a_pn, Dg)
    ok = np.isfinite(s_gt) & np.isfinite(s_pn)
    print("\n1. độ nghiêng biên dα*/dD  (công thức cân bằng cũ cho +19.74)")
    for lab, m in (("β < 1.5", be < 1.5), ("β ≥ 1.5", be >= 1.5), ("tất cả", be > -1)):
        mm = ok & m
        if mm.sum() == 0:
            continue
        agree = np.mean(np.sign(s_gt[mm]) == np.sign(s_pn[mm]))
        print(f"    {lab:<8s} thật: trung vị {np.median(s_gt[mm]):+7.3f}   "
              f"PINN: trung vị {np.median(s_pn[mm]):+7.3f}   cùng dấu {agree:.0%}  (n={mm.sum()})")

    # ── 2. dịch ngang hay sai hình dạng ───────────────────────────────────────
    d = a_pn - a_gt
    print("\n2. Δα = α*_PINN − α*_thật theo lát D  (dương = PINN đặt biên lệch phải, tức báo an toàn muộn)")
    print(f"    {'D':>8s} {'trung vị Δα':>12s} {'p10..p90 theo β':>18s} {'n':>4s}")
    for k in range(0, len(Dg), 3):
        v = d[:, k][np.isfinite(d[:, k])]
        if len(v) >= 3:
            print(f"    {Dg[k]:8.4f} {np.median(v):+12.3f} "
                  f"{np.percentile(v,10):+8.3f}..{np.percentile(v,90):+7.3f} {len(v):4d}")

    B, Dm = np.meshgrid(be, Dg, indexing="ij")
    m = np.isfinite(d)
    y = d[m]
    X = np.column_stack([np.ones(m.sum()), B[m], Dm[m]])
    res_lin = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    per_slice = np.array([np.nanmedian(d[:, k]) for k in range(len(Dg))])
    res_sl = (d - per_slice[None, :])[m]
    print(f"\n    |Δα| trung vị, chuẩn hoá theo miền α:")
    print(f"      nguyên bản                      {np.median(np.abs(y))/span_a:.4f}")
    print(f"      bỏ một hằng số cho mỗi lát D    {np.median(np.abs(res_sl))/span_a:.4f}")
    print(f"      bỏ một hàm tuyến tính theo (β,D) {np.median(np.abs(res_lin))/span_a:.4f}")
    print(f"    (để so: δ_H p95 của B2 = 0.053, BO = 0.034, sàn phép đo = 0.038)")