"""
plot_compare.py — bản đồ pha: biên thật, biên FMD-PINN, biên BO, trên cùng trục.

  python plot_compare.py --tag cap
  python plot_compare.py --tag cap --bo bo_boundary --variants ls1_bs0_kn0

Ba lát D. Đường đen liền là biên thật từ FEM. Các đường màu là biên dự đoán,
tức tập mức không của từng mô hình thay thế. Chữ thập là các lời gọi oracle
của biến thể FMD được chọn, đỏ là sập, xanh là ổn định.

Biên BO dựng đúng như trong metrics: khớp một GP Matérn trên lịch sử oracle
của chính BO rồi lấy đường mức không của nó. Không phải nội suy điểm.
"""
import sys, os, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="cap")
ap.add_argument("--seed", type=int, default=cfg.BASE_SEED)
ap.add_argument("--hidden", type=int, default=None)
ap.add_argument("--fourier", type=int, default=None)
ap.add_argument("--bo", default="bo_boundary",
                choices=["bo_boundary", "bo_fem", "none"],
                help="lịch sử BO nào dùng để dựng biên GP")
ap.add_argument("--variants", default=None,
                help="danh sách cờ FMD, ngăn bằng dấu phẩy, ví dụ ls1_bs0_kn0")
ap.add_argument("--calls-from", default=None,
                help="vẽ điểm gọi oracle của cờ này, ví dụ ls1_bs0_kn0")
args = ap.parse_args()

if args.hidden:
    cfg.SPATIAL_HIDDEN = [args.hidden] * len(cfg.SPATIAL_HIDDEN)
if args.fourier:
    cfg.N_FOURIER = args.fourier

import torch
from src.ground_truth import generate_ground_truth
from src.pinn_model import build_model, normalise_params
from src.loss_functions import failure_functional
from src.metrics import bo_style_surrogate

ALL = [
    ("ls0_bs0_kn0", "B1 thụ động",      "tab:gray",   "-."),
    ("ls1_bs0_kn0", "B2 tập mức không", "tab:blue",   "--"),
    ("ls1_bs1_kn0", "B3 + bisection",   "tab:green",  ":"),
    ("ls1_bs1_kn1", "B4 + knee",        "tab:orange", "-"),
]
if args.variants:
    keep = {v.strip() for v in args.variants.split(",")}
    ALL = [v for v in ALL if v[0] in keep]

gt = generate_ground_truth()
al, be, Dg = gt["alpha_grid"], gt["beta_grid"], gt["D_grid"]
AG, BG = np.meshgrid(al, be, indexing="ij")

NA = NB = 110
Af, Bf = np.meshgrid(np.linspace(*cfg.ALPHA_RANGE, NA),
                     np.linspace(*cfg.BETA_RANGE, NB), indexing="ij")


def pinn_slice(model, D_val):
    flat = np.column_stack([Af.ravel(), Bf.ravel(), np.full(Af.size, D_val)])
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), 256):
            b = flat[i:i + 256]
            p = normalise_params(
                torch.tensor(b[:, 0], dtype=torch.float32, device=cfg.DEVICE),
                torch.tensor(b[:, 1], dtype=torch.float32, device=cfg.DEVICE),
                torch.tensor(b[:, 2], dtype=torch.float32, device=cfg.DEVICE))
            out.append(failure_functional(model, p, device=cfg.DEVICE).cpu().numpy())
    return np.concatenate(out).reshape(Af.shape)


def bo_slice(history, D_val):
    """Đường mức không của GP khớp trên lịch sử oracle của BO."""
    flat = np.column_stack([Af.ravel(), Bf.ravel(), np.full(Af.size, D_val)])
    return bo_style_surrogate(history, flat).reshape(Af.shape)


# ── nạp mô hình FMD ──────────────────────────────────────────────────────────
loaded = []
for flags, label, color, ls in ALL:
    name = f"fmd_pinn_seed{args.seed}_{flags}" + (f"_{args.tag}" if args.tag else "")
    path = os.path.join(cfg.CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        cands = [f for f in os.listdir(cfg.CKPT_DIR)
                 if f.startswith(f"{name}_call") and f.endswith(".pt")]
        if not cands:
            print(f"  bỏ qua {label}: không thấy checkpoint cho {name}")
            continue
        path = os.path.join(cfg.CKPT_DIR, max(
            cands, key=lambda f: int(f.rsplit("call", 1)[1][:-3])))
    m = build_model(cfg.DEVICE)
    m.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
    m.eval()
    loaded.append((m, label, color, ls, flags))

# ── nạp lịch sử BO ───────────────────────────────────────────────────────────
bo_hist = None
if args.bo != "none":
    p = os.path.join(cfg.RESULTS_DIR, f"{args.bo}_seed{args.seed}.json")
    if os.path.exists(p):
        bo_hist = json.load(open(p))["oracle_history"]
        n_edge = sum(1 for r in bo_hist if abs(r["E_true"]) < 0.3)
        print(f"  BO: {len(bo_hist)} lời gọi, {n_edge} nằm trong dải biên")
    else:
        print(f"  không thấy {os.path.basename(p)}, bỏ phần BO")

# ── điểm gọi oracle của một biến thể FMD ─────────────────────────────────────
calls = []
cf = args.calls_from or (loaded[0][4] if loaded else None)
if cf:
    hp = os.path.join(cfg.RESULTS_DIR,
                      f"fmd_pinn_seed{args.seed}_{cf}"
                      + (f"_{args.tag}" if args.tag else "") + ".json")
    if os.path.exists(hp):
        calls = json.load(open(hp)).get("oracle_history", [])

idx = [1, len(Dg) // 2, len(Dg) - 2]
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), sharey=True)

for col, k in enumerate(idx):
    ax, D_val = axes[col], float(Dg[k])
    Et = gt["E_fem"][:, :, k]
    v = max(abs(Et.min()), abs(Et.max()), 0.1)
    ax.contourf(AG, BG, Et, levels=40, cmap="RdBu_r", vmin=-v, vmax=v, alpha=0.30)
    ax.contour(AG, BG, Et, levels=[0.0], colors="k", linewidths=3.0)

    for m, label, color, ls, _ in loaded:
        ax.contour(Af, Bf, pinn_slice(m, D_val), levels=[0.0],
                   colors=color, linewidths=2.0, linestyles=ls)

    if bo_hist is not None:
        ax.contour(Af, Bf, bo_slice(bo_hist, D_val), levels=[0.0],
                   colors="tab:purple", linewidths=2.0, linestyles="-.")

    pts = [r for r in calls if abs(np.log(r["D"]) - np.log(D_val)) < 0.45]
    if pts:
        ax.scatter([r["alpha"] for r in pts], [r["beta"] for r in pts],
                   c=["#b2182b" if r["E_true"] > 0 else "#2166ac" for r in pts],
                   marker="x", s=26, linewidths=1.1, zorder=5)

    ax.axhline(cfg.KNEE_BETA, color="0.35", ls=":", lw=1)
    ax.set_title(f"D = {D_val:.4f}"
                 + (f"   ({len(pts)} lời gọi gần lát)" if pts else ""), fontsize=11)
    ax.set_xlabel("α")
axes[0].set_ylabel("β")

handles = [plt.Line2D([0], [0], color="k", lw=3, label="biên thật (FEM)")]
handles += [plt.Line2D([0], [0], color=c, lw=2, ls=l, label=lab)
            for _, lab, c, l, _ in loaded]
if bo_hist is not None:
    handles.append(plt.Line2D([0], [0], color="tab:purple", lw=2,
                              ls=(0, (4, 2, 1, 2)), label="BO (GP, tìm biên)"))
handles.append(plt.Line2D([0], [0], color="0.35", lw=1, ls=":",
                          label=f"β = {cfg.KNEE_BETA} (ranh giới vùng knee)"))
axes[0].legend(handles=handles, loc="upper left", fontsize=8)

npar = sum(p.numel() for p in loaded[0][0].parameters()) if loaded else 0
fig.suptitle(f"Biên dự đoán so với biên thật   (seed {args.seed}"
             + (f", PINN {npar:,} tham số" if npar else "") + ")", fontsize=12)
fig.tight_layout()
out = os.path.join(cfg.RESULTS_DIR,
                   f"compare_boundary{'_' + args.tag if args.tag else ''}.png")
fig.savefig(out, dpi=180, facecolor="white")
print(f"✓ {out}")