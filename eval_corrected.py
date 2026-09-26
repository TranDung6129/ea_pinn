"""
eval_corrected.py — PINN cho hình dạng, oracle cho mức. Không huấn luyện lại.

  python eval_corrected.py --tag cap
  python eval_corrected.py --tag cap --variants ls1_bs0_kn0,ls1_bs1_kn0

E_final(p) = E_pinn(p) + r(p), r khớp trên phần dư E_true − E_pinn tại CHÍNH
200 lời gọi oracle của phương pháp (không dùng ground truth). Bốn cách chọn r:

  none   r = 0                        kiểm tra: phải ra lại δ_H cũ của PINN
  const  r = hằng số                  trung bình phần dư trong dải |E_true| < 0.3
  logD2  r = c0 + c1·u + c2·u²        u = log D chuẩn hoá; 3 tham số, cũng trong dải biên
  gp     r = GP trên phần dư          CÙNG cấu hình GP với BO (Matérn 2.5, cùng đầu vào),
                                      nên khác BO đúng một thứ: hàm trung bình là PINN
                                      thay vì hằng số (Kennedy & O'Hagan 2001)

Đo bằng đúng các hàm BO dùng: trích biên trên lưới đo 24³, FSR/CR trên lưới GT.
Chỉ số cuối cùng (sau 200 lời gọi); không báo N_δ vì mô hình là bản cuối.

Ngưỡng chốt trước khi chạy: thành công nếu δ_H p95 trung bình 5 seed ≤ 0.040
và FSR_biên không tệ hơn bản không hiệu chỉnh.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import config as cfg
from src.ground_truth import generate_ground_truth
from src.pinn_model import build_model
from src.metrics import (metric_grid_params, extract_boundary_from_E, boundary_distances,
                         grid_safety_metrics, normalise_for_hausdorff, pinn_E_on_grid,
                         bo_style_surrogate)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="cap")
ap.add_argument("--variants", default="ls1_bs0_kn0,ls1_bs1_kn0")
ap.add_argument("--seeds", type=int, default=cfg.N_SEEDS)
ap.add_argument("--band", type=float, default=0.3, help="dải |E_true| dùng để khớp const và logD2")
args = ap.parse_args()
dev = cfg.DEVICE
METHODS = ["none", "const", "logD2", "gp"]


def load_model(flags, phys):
    name = f"fmd_pinn_seed{phys}_{flags}" + (f"_{args.tag}" if args.tag else "")
    path = os.path.join(cfg.CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        cands = [f for f in os.listdir(cfg.CKPT_DIR)
                 if f.startswith(f"{name}_call") and f.endswith(".pt")]
        if not cands:
            return None, None
        path = os.path.join(cfg.CKPT_DIR,
                            max(cands, key=lambda f: int(f.rsplit("call", 1)[1][:-3])))
    m = build_model(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    hp = os.path.join(cfg.RESULTS_DIR, f"{name}.json")
    hist = json.load(open(hp))["oracle_history"] if os.path.exists(hp) else None
    return m, hist


def u_logD(D):
    return ((np.log(D) - np.log(cfg.D_RANGE[0])) /
            (np.log(cfg.D_RANGE[1]) - np.log(cfg.D_RANGE[0])))


def fit_correction(method, X, res, E_true):
    """Trả về hàm r(params). X, res, E_true lấy từ lịch sử oracle của chính phương pháp."""
    if method == "none":
        return lambda P: np.zeros(len(P))
    near = np.abs(E_true) < args.band
    if near.sum() < 5:
        near = np.ones_like(near)
    if method == "const":
        c = float(np.mean(res[near]))
        return lambda P: np.full(len(P), c)
    if method == "logD2":
        u = u_logD(X[near, 2])
        c = np.linalg.lstsq(np.column_stack([np.ones_like(u), u, u * u]), res[near], rcond=None)[0]
        return lambda P: c[0] + c[1] * u_logD(P[:, 2]) + c[2] * u_logD(P[:, 2]) ** 2
    if method == "gp":
        # bo_style_surrogate khớp GP lên cột E_true; ở đây đưa phần dư vào cột đó
        h = [dict(alpha=a, beta=b, D=d, E_true=r) for (a, b, d), r in zip(X, res)]
        return lambda P: bo_style_surrogate(h, P)
    raise ValueError(method)


def evaluate(E_mp, E_gt, mp, gt, true_b):
    d = boundary_distances(extract_boundary_from_E(mp, E_mp), true_b)
    g = grid_safety_metrics(None, gt, E_pred=E_gt)
    return dict(p95=d["p95"], chamfer=d["chamfer"], fsr=g["fsr_band"],
                far=g["far_band"], cr=g["cr"], bias=g["bias_band"])


def show(label, rows):
    if not rows:
        return
    k = ["p95", "chamfer", "fsr", "far", "cr", "bias"]
    a = {x: np.array([r[x] for r in rows]) for x in k}
    print(f"  {label:<18s} {a['p95'].mean():.4f}±{a['p95'].std():.4f}  "
          f"{a['chamfer'].mean():.4f}  {a['fsr'].mean():6.1%}  {a['far'].mean():6.1%}  "
          f"{a['cr'].mean():6.1%}  {a['bias'].mean():+.3f}   (n={len(rows)})")


if __name__ == "__main__":
    gt = generate_ground_truth()
    mask_gt = np.abs(gt["E_fem_flat"]) < 0.15
    P = gt["params_flat"]
    true_b = normalise_for_hausdorff(P[mask_gt, 0], P[mask_gt, 1], P[mask_gt, 2])
    mp = metric_grid_params(cfg.METRIC_GRID)
    hdr = (f"  {'':<18s} {'δ_H p95':>13s}  {'chamfer':>7s}  {'FSR_b':>6s}  {'BĐgiả':>6s}  "
           f"{'CR':>6s}  {'lệch':>6s}")

    for flags in [v.strip() for v in args.variants.split(",")]:
        rows = {m: [] for m in METHODS}
        for s in range(args.seeds):
            phys = cfg.BASE_SEED + s
            model, hist = load_model(flags, phys)
            if model is None or hist is None:
                print(f"  bỏ qua {flags} seed {phys}: thiếu checkpoint hoặc lịch sử")
                continue
            X = np.array([[r["alpha"], r["beta"], r["D"]] for r in hist])
            y = np.array([r["E_true"] for r in hist])
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            X, y = X[ok], y[ok]
            E_hist = pinn_E_on_grid(model, X, dev)          # mô hình CUỐI, không dùng E_pinn lưu lúc gọi
            E_mp0, E_gt0 = pinn_E_on_grid(model, mp, dev), pinn_E_on_grid(model, P, dev)
            res = y - E_hist
            for m in METHODS:
                r = fit_correction(m, X, res, y)
                rows[m].append(evaluate(E_mp0 + r(mp), E_gt0 + r(P), mp, gt, true_b))
            print(f"  {flags} seed {phys}: phần dư trong dải biên trung vị "
                  f"{np.median(res[np.abs(y) < args.band]):+.4f}   "
                  f"p95 none {rows['none'][-1]['p95']:.4f} → gp {rows['gp'][-1]['p95']:.4f}")
        print(f"\n── {flags} " + "─" * 60)
        print(hdr)
        for m in METHODS:
            show(m, rows[m])

    # BO tham chiếu, chấm bằng đúng pipeline trên. Kiểm tra vùng: lần chạy holdout
    # (--region beta:2.3..5) ghi đè cùng tên file.
    bo = []
    for s in range(args.seeds):
        p = os.path.join(cfg.RESULTS_DIR, f"bo_boundary_seed{cfg.BASE_SEED + s}.json")
        if not os.path.exists(p):
            continue
        h = json.load(open(p))["oracle_history"]
        if min(r["beta"] for r in h) > 2.0:
            print(f"\n  ⚠ {os.path.basename(p)} là bản chạy holdout (β chỉ từ "
                  f"{min(r['beta'] for r in h):.2f}), không dùng làm tham chiếu."
                  f" Chạy lại: python run_03_bayesian_opt.py --which boundary --force")
            bo = []
            break
        bo.append(evaluate(bo_style_surrogate(h, mp), bo_style_surrogate(h, P), mp, gt, true_b))
    if bo:
        print("\n── BO tìm biên, cùng pipeline " + "─" * 40)
        print(hdr)
        show("bo_boundary", bo)