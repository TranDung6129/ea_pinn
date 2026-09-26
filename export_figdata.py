"""
export_figdata.py — gom mọi dữ liệu cần để vẽ hình cho bài báo vào MỘT file.
Không huấn luyện gì. Chạy khoảng 2 đến 3 phút (phần lớn là 1500 lời gọi FEM cho gradient).

  python export_figdata.py
  → results/figdata.npz   (gửi file này)

Gồm:
  - ground truth 25³
  - E_pinn trên 3 lát D (lưới 110×110) cho B1/B2/B3 × tag cap, proj × 5 seed, và b (hiệu chỉnh hằng số)
  - E của GP BO trên cùng các lát, 5 seed
  - lịch sử oracle của mọi lần chạy, và metrics theo seed (gồm đường δ_H theo số lời gọi)
  - kiểm gradient cho B3 `cap`, 5 seed: 50 điểm trên biên thật, autodiff và sai phân trung tâm trên FEM
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import config as cfg
from src.ground_truth import generate_ground_truth
from src.pinn_model import build_model, normalise_params
from src.loss_functions import failure_functional
from src.metrics import pinn_E_on_grid, bo_style_surrogate
from src.fem_oracle import batch_oracle

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="cap,proj")
ap.add_argument("--seeds", type=int, default=cfg.N_SEEDS)
ap.add_argument("--n-slice", type=int, default=110)
ap.add_argument("--n-grad", type=int, default=50)
ap.add_argument("--h", type=float, default=0.02)
ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR, "figdata.npz"))
args = ap.parse_args()
dev = cfg.DEVICE
VARIANTS = {"ls0_bs0_kn0": "B1_passive", "ls1_bs0_kn0": "B2_levelset", "ls1_bs1_kn0": "B3_bisect"}
KEEP = ["hausdorff_calls", "hausdorff_curve", "dh_p95", "dh_chamfer", "dh_max", "fsr_band",
        "far_band", "coverage_ratio", "bias_band", "fsr_knee", "far_knee", "n_delta"]


def ckpt(flags, phys, tag):
    name = f"fmd_pinn_seed{phys}_{flags}_{tag}"
    path = os.path.join(cfg.CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        c = [f for f in os.listdir(cfg.CKPT_DIR) if f.startswith(f"{name}_call") and f.endswith(".pt")]
        if not c:
            return None, None
        path = os.path.join(cfg.CKPT_DIR, max(c, key=lambda f: int(f.rsplit("call", 1)[1][:-3])))
    hp = os.path.join(cfg.RESULTS_DIR, f"{name}.json")
    if not os.path.exists(hp):
        return None, None
    m = build_model(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    return m, json.load(open(hp))["oracle_history"]


def slim(h):
    k = ("alpha", "beta", "D", "E_true", "E_pinn", "kind")
    return [{x: r[x] for x in k if x in r} for r in h]


def grad_ad(model, pts):
    t = [torch.tensor(pts[:, i], dtype=torch.float32, device=dev, requires_grad=True) for i in range(3)]
    E = failure_functional(model, normalise_params(*t), device=dev)
    g = torch.autograd.grad(E.sum(), t)
    return E.detach().cpu().numpy(), torch.stack(g, 1).cpu().numpy()


def boundary_pts(gt, n, h_abs, rng):
    al, be, Dg, E = gt["alpha_grid"], gt["beta_grid"], gt["D_grid"], gt["E_fem"]
    lo = np.array([al.min(), be.min(), Dg.min()]) + h_abs
    hi = np.array([al.max(), be.max(), Dg.max()]) - h_abs
    out = []
    for j in range(len(be)):
        for k in range(len(Dg)):
            c = E[:, j, k]
            for i in np.where(np.sign(c[:-1]) * np.sign(c[1:]) < 0)[0]:
                t = c[i] / (c[i] - c[i + 1])
                p = np.array([al[i] + t * (al[i + 1] - al[i]), be[j], Dg[k]])
                if np.all(p > lo) and np.all(p < hi):
                    out.append(p)
    out = np.array(out)
    return out[rng.choice(len(out), size=min(n, len(out)), replace=False)]


if __name__ == "__main__":
    gt = generate_ground_truth()
    Dg = gt["D_grid"]
    sD = np.array([Dg[1], Dg[len(Dg) // 2], Dg[-2]])
    sa = np.linspace(*cfg.ALPHA_RANGE, args.n_slice)
    sb = np.linspace(*cfg.BETA_RANGE, args.n_slice)
    A, B = np.meshgrid(sa, sb, indexing="ij")
    flat = np.concatenate([np.column_stack([A.ravel(), B.ravel(), np.full(A.size, d)]) for d in sD])
    shp = (len(sD), args.n_slice, args.n_slice)

    arr = dict(gt_alpha=gt["alpha_grid"], gt_beta=gt["beta_grid"], gt_D=Dg, gt_E=gt["E_fem"],
               slice_D=sD, slice_alpha=sa, slice_beta=sb)
    meta = dict(fmd={}, bo={}, metrics={}, b={}, config=dict(
        alpha_range=cfg.ALPHA_RANGE, beta_range=cfg.BETA_RANGE, D_range=cfg.D_RANGE,
        gt_grid=int(gt["n_grid"]), metric_grid=cfg.METRIC_GRID, budget=cfg.ORACLE_BUDGET,
        h_grad=args.h))

    # ── FMD: lát cắt, b, lịch sử, metrics ────────────────────────────────────
    for tag in args.tags.split(","):
        for flags, label in VARIANTS.items():
            for s in range(args.seeds):
                phys = cfg.BASE_SEED + s
                m, h = ckpt(flags, phys, tag)
                if m is None:
                    continue
                key = f"{tag}_{flags}_{phys}"
                X = np.array([[r["alpha"], r["beta"], r["D"]] for r in h])
                y = np.array([r["E_true"] for r in h])
                near = np.abs(y) < 0.3
                b = float(np.mean(y[near] - pinn_E_on_grid(m, X[near], dev))) if near.sum() >= 3 else 0.0
                arr[f"E_{key}"] = pinn_E_on_grid(m, flat, dev).reshape(shp).astype(np.float32)
                meta["b"][key] = b
                meta["fmd"][key] = slim(h)
                mp = os.path.join(cfg.RESULTS_DIR, f"{label}_{tag}_seed{s}_metrics.json")
                if os.path.exists(mp):
                    mj = json.load(open(mp))
                    meta["metrics"][key] = {k: mj[k] for k in KEEP if k in mj}
                print(f"  {key}: b={b:+.4f}")

    # ── BO ───────────────────────────────────────────────────────────────────
    for s in range(args.seeds):
        phys = cfg.BASE_SEED + s
        p = os.path.join(cfg.RESULTS_DIR, f"bo_boundary_seed{phys}.json")
        if not os.path.exists(p):
            continue
        h = json.load(open(p))["oracle_history"]
        if min(r["beta"] for r in h) > 2.0:
            print(f"  ⚠ {os.path.basename(p)} là bản holdout, bỏ qua")
            continue
        arr[f"E_bo_{phys}"] = bo_style_surrogate(h, flat).reshape(shp).astype(np.float32)
        meta["bo"][str(phys)] = slim(h)
        print(f"  bo {phys}")

    # ── gradient: B3 cap, mỗi seed 50 điểm trên biên thật ────────────────────
    span = np.array([cfg.ALPHA_RANGE[1] - cfg.ALPHA_RANGE[0],
                     cfg.BETA_RANGE[1] - cfg.BETA_RANGE[0], Dg.max() - Dg.min()])
    h_abs = args.h * span
    for s in range(args.seeds):
        phys = cfg.BASE_SEED + s
        m, _ = ckpt("ls1_bs1_kn0", phys, "cap")
        if m is None:
            continue
        pts = boundary_pts(gt, args.n_grad, h_abs, np.random.default_rng(phys))
        E_ad, g_ad = grad_ad(m, pts)
        qs = []
        for i in range(3):
            for sg in (1.0, -1.0):
                q = pts.copy(); q[:, i] += sg * h_abs[i]; qs.append(q)
        Ef = np.asarray(batch_oracle(np.vstack(qs), show_progress=False), dtype=float).reshape(6, len(pts))
        g_fem = np.stack([(Ef[2 * i] - Ef[2 * i + 1]) / (2 * h_abs[i]) for i in range(3)], 1)
        arr[f"grad_{phys}_pts"], arr[f"grad_{phys}_ad"] = pts, g_ad
        arr[f"grad_{phys}_fem"], arr[f"grad_{phys}_E"] = g_fem, E_ad
        a, c = g_fem * span, g_ad * span
        cos = (a * c).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1) + 1e-30)
        print(f"  gradient {phys}: cos trung vị {np.median(cos):.4f}")

    arr["meta_json"] = np.array(json.dumps(meta, default=float))
    np.savez_compressed(args.out, **arr)
    print(f"\n✓ {args.out}  ({os.path.getsize(args.out)/2**20:.1f} MB)")