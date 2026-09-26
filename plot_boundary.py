"""
plot_boundary.py — nhìn biên dự đoán so với biên thật.

  python plot_boundary.py                                   # mạng mặc định
  python plot_boundary.py --tag cap --hidden 128 --fourier 64

Ba lát D. Hàng trên: nền là E_thật, đường đen là biên thật, đường màu là biên
dự đoán, chữ thập là các điểm đã gọi oracle gần lát đó. Hàng dưới: bản đồ
lệch E_pinn - E_true, đỏ là mô hình đoán nguy hiểm hơn thực tế, xanh là đoán
an toàn hơn thực tế (tức phía gây false-safe).
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=cfg.BASE_SEED)
ap.add_argument("--hidden", type=int, default=None)
ap.add_argument("--layers", type=int, default=None)
ap.add_argument("--fourier", type=int, default=None)
ap.add_argument("--flags", default="ls1_bs1_kn1")
args = ap.parse_args()

if args.hidden or args.layers:
    cfg.SPATIAL_HIDDEN = [args.hidden or cfg.SPATIAL_HIDDEN[0]] * \
                         (args.layers or len(cfg.SPATIAL_HIDDEN))
if args.fourier:
    cfg.N_FOURIER = args.fourier

import torch
from src.ground_truth import generate_ground_truth
from src.pinn_model import build_model, normalise_params
from src.loss_functions import failure_functional

name = f"fmd_pinn_seed{args.seed}_{args.flags}" + (f"_{args.tag}" if args.tag else "")
ckpt = os.path.join(cfg.CKPT_DIR, f"{name}.pt")
hist_file = os.path.join(cfg.RESULTS_DIR, f"{name}.json")
if not os.path.exists(ckpt):
    cands = [f for f in os.listdir(cfg.CKPT_DIR)
             if f.startswith(f"{name}_call") and f.endswith(".pt")]
    if not cands:
        sys.exit(f"Không thấy checkpoint nào cho {name}")
    ckpt = os.path.join(cfg.CKPT_DIR, max(
        cands, key=lambda f: int(f.rsplit("call", 1)[1][:-3])))
    print(f"  dùng {os.path.basename(ckpt)}")

gt = generate_ground_truth()
al, be, Dg = gt["alpha_grid"], gt["beta_grid"], gt["D_grid"]
E_true_3d = gt["E_fem"]

model = build_model(cfg.DEVICE)
model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))
model.eval()

history = []
if os.path.exists(hist_file):
    history = json.load(open(hist_file)).get("oracle_history", [])


def E_pred_slice(D_val, na=80, nb=80):
    """E_pinn trên lưới mịn (alpha,beta) tại D cố định."""
    A, B = np.meshgrid(np.linspace(*cfg.ALPHA_RANGE, na),
                       np.linspace(*cfg.BETA_RANGE, nb), indexing="ij")
    flat = np.column_stack([A.ravel(), B.ravel(),
                            np.full(A.size, D_val)])
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), 256):
            b = flat[i:i + 256]
            p = normalise_params(
                torch.tensor(b[:, 0], dtype=torch.float32, device=cfg.DEVICE),
                torch.tensor(b[:, 1], dtype=torch.float32, device=cfg.DEVICE),
                torch.tensor(b[:, 2], dtype=torch.float32, device=cfg.DEVICE))
            out.append(failure_functional(model, p, device=cfg.DEVICE).cpu().numpy())
    return A, B, np.concatenate(out).reshape(A.shape)


def E_pred_on_gt_slice(k):
    """E_pinn tại đúng các nút lưới GT của lát D thứ k, để tính lệch."""
    A, B = np.meshgrid(al, be, indexing="ij")
    flat = np.column_stack([A.ravel(), B.ravel(), np.full(A.size, Dg[k])])
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), 256):
            b = flat[i:i + 256]
            p = normalise_params(
                torch.tensor(b[:, 0], dtype=torch.float32, device=cfg.DEVICE),
                torch.tensor(b[:, 1], dtype=torch.float32, device=cfg.DEVICE),
                torch.tensor(b[:, 2], dtype=torch.float32, device=cfg.DEVICE))
            out.append(failure_functional(model, p, device=cfg.DEVICE).cpu().numpy())
    return A, B, np.concatenate(out).reshape(A.shape)


idx = [1, len(Dg) // 2, len(Dg) - 2]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for col, k in enumerate(idx):
    D_val = float(Dg[k])
    AG, BG = np.meshgrid(al, be, indexing="ij")
    Et = E_true_3d[:, :, k]

    # ── hàng trên: biên thật vs dự đoán ──
    ax = axes[0, col]
    v = max(abs(Et.min()), abs(Et.max()), 0.1)
    ax.contourf(AG, BG, Et, levels=40, cmap="RdBu_r", vmin=-v, vmax=v, alpha=0.5)
    ax.contour(AG, BG, Et, levels=[0.0], colors="k", linewidths=2.5)

    Af, Bf, Ef = E_pred_slice(D_val)
    ax.contour(Af, Bf, Ef, levels=[0.0], colors="lime", linewidths=2.0,
               linestyles="--")

    pts = [r for r in history
           if abs(np.log(r["D"]) - np.log(D_val)) < 0.45]
    if pts:
        ax.scatter([r["alpha"] for r in pts], [r["beta"] for r in pts],
                   c=["#b2182b" if r["E_true"] > 0 else "#2166ac" for r in pts],
                   marker="x", s=30, linewidths=1.2, zorder=5)
    ax.set_title(f"D = {D_val:.4f}   ({len(pts)} lời gọi gần lát)", fontsize=10)
    ax.set_xlabel("α"); ax.set_ylabel("β")

    # ── hàng dưới: lệch ──
    ax = axes[1, col]
    _, _, Ep = E_pred_on_gt_slice(k)
    diff = Ep - Et
    m = max(abs(diff).max(), 1e-3)
    im = ax.pcolormesh(AG, BG, diff, cmap="RdBu_r", vmin=-m, vmax=m,
                       shading="auto")
    ax.contour(AG, BG, Et, levels=[0.0], colors="k", linewidths=2.0)
    plt.colorbar(im, ax=ax, label="E_pinn − E_true")
    ax.set_title(f"lệch trung bình {diff.mean():+.3f}", fontsize=10)
    ax.set_xlabel("α"); ax.set_ylabel("β")

axes[0, 0].legend(handles=[
    plt.Line2D([0], [0], color="k", lw=2.5, label="biên thật (FEM)"),
    plt.Line2D([0], [0], color="lime", lw=2.0, ls="--", label="biên dự đoán"),
    plt.Line2D([0], [0], color="#b2182b", marker="x", ls="", label="gọi: sập"),
    plt.Line2D([0], [0], color="#2166ac", marker="x", ls="", label="gọi: an toàn"),
], loc="upper right", fontsize=7)

fig.suptitle(f"{name}   ({sum(p.numel() for p in model.parameters()):,} tham số)",
             fontsize=11)
fig.tight_layout()
out = os.path.join(cfg.RESULTS_DIR, f"boundary_{name}.png")
fig.savefig(out, dpi=180, facecolor="white")
print(f"✓ {out}")