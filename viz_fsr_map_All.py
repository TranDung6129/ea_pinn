"""
viz_fsr_map_all.py — false-safe locations for ALL seeds, side by side.

2x3 grid: one panel per seed (5 seeds + 1 summary panel).
Each panel: true boundary contour + correct calls (grey) +
false-safe calls (red X, colored by call#).
Lets you eyeball why seed0/1 have low FSR while seed3/4 are high.

Usage: python viz_fsr_map_all.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg

SUFFIX = "minmax1_ev1_as1"
TAU    = 0.05
DPI    = 200
SEEDS  = [cfg.BASE_SEED + i for i in range(cfg.N_SEEDS)]   # 42..46

# ── Load GT contour once ──
gt = np.load(os.path.join(cfg.RESULTS_DIR, "ground_truth.npz"))
print("GT keys:", list(gt.keys()))

def draw_gt(ax):
    for ak, bk, ek in [("alpha_grid","beta_grid","E_grid"),
                       ("alphas","betas","E"),
                       ("A","B","E")]:
        if ak in gt and bk in gt and ek in gt:
            A, B, E = gt[ak], gt[bk], gt[ek]
            E2 = E[:, :, E.shape[2]//2] if E.ndim == 3 else E
            try:
                ax.contourf(A, B, E2.T, levels=30, cmap="RdBu_r", alpha=0.30)
                ax.contour(A, B, E2.T, levels=[0.0], colors="k", linewidths=2.0)
            except Exception:
                ax.contourf(A, B, E2, levels=30, cmap="RdBu_r", alpha=0.30)
                ax.contour(A, B, E2, levels=[0.0], colors="k", linewidths=2.0)
            return True
    return False

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
axes = axes.ravel()

summary = []
for i, seed in enumerate(SEEDS):
    ax = axes[i]
    hist_file = os.path.join(cfg.RESULTS_DIR, f"fmd_pinn_seed{seed}_{SUFFIX}.json")
    if not os.path.exists(hist_file):
        ax.set_title(f"seed idx {i} (s{seed}) — missing")
        continue
    with open(hist_file) as f:
        hist = json.load(f)["oracle_history"]

    draw_gt(ax)

    ca, cb, fa, fb, fc = [], [], [], [], []
    for r in hist:
        a, b = r["alpha"], r["beta"]
        ep, et = r.get("E_pinn", -1), r.get("E_true", 0)
        if ep < 0 and et > TAU:
            fa.append(a); fb.append(b); fc.append(r.get("oracle_call", 0))
        else:
            ca.append(a); cb.append(b)

    ax.scatter(ca, cb, c="grey", s=12, alpha=0.35)
    if fa:
        ax.scatter(fa, fb, c=fc, cmap="autumn_r", s=70, marker="X",
                   edgecolors="k", linewidths=0.5, zorder=5)
    fsr = len(fa) / max(1, sum(1 for r in hist if r.get("E_true",0) > TAU))
    knee = sum(1 for b in fb if b < 1.5)
    knee_pct = 100*knee/len(fa) if fa else 0
    ax.set_title(f"seed idx {i} (s{seed})  FSR={fsr*100:.1f}%  "
                 f"n_FS={len(fa)}  knee={knee_pct:.0f}%")
    ax.set_xlabel("α"); ax.set_ylabel("β")
    summary.append((i, seed, fsr*100, len(fa), knee_pct))

# last panel: textual summary
axes[5].axis("off")
txt = "Seed   FSR    n_FS  knee%\n" + "-"*30 + "\n"
for i, s, fsr, n, kp in summary:
    txt += f"idx{i} s{s}  {fsr:5.1f}%  {n:3d}  {kp:4.0f}%\n"
axes[5].text(0.05, 0.95, txt, family="monospace", fontsize=12,
             va="top", transform=axes[5].transAxes)

fig.suptitle(f"False-Safe Locations by Seed  (τ={TAU})  "
             f"— X = false-safe, grey = correct", fontsize=14)
fig.tight_layout()
out = os.path.join(cfg.RESULTS_DIR, "fsr_map_all_seeds.png")
fig.savefig(out, dpi=DPI, facecolor="white")
print(f"✓ Saved {out}")

print("\nSummary:")
for i, s, fsr, n, kp in summary:
    print(f"  seed idx {i} (s{s}): FSR={fsr:.1f}%  n_FS={n}  in_knee(β<1.5)={kp:.0f}%")