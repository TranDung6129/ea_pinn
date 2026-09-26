"""
diag_knee.py — vùng knee là gì: hình học hay đổi cơ chế hỏng?

Giả thuyết cần kiểm: biên giải tích trong src/ground_truth.py,
    u* = sqrt((alpha - 2*pi^2*D) / beta),  biên là u* = u_thr,
xuất phát từ BIÊN ĐỘ CÂN BẰNG, tức trạng thái dừng. Nó khớp dấu với FEM
90.7% trên toàn lưới. Nếu 9.3% sai đó tụm đúng vào beta thấp, và nếu ở đó u
đạt đỉnh GIỮA khoảng thời gian thay vì ở cuối, thì knee không phải chỗ mô
hình học kém mà là chỗ CƠ CHẾ HỎNG ĐỔI: hệ vượt ngưỡng bằng quá độ trong
thời gian hữu hạn chứ không bằng mức cân bằng.

Chạy:  python diag_knee.py
Đọc từ ground truth có sẵn + một ít lời gọi FEM để lấy quỹ đạo theo thời gian.
Không huấn luyện lại gì.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
from src.ground_truth import generate_ground_truth, analytical_failure
from src.fem_oracle import solve_reaction_diffusion

NG = 15          # lưới (alpha, beta) cho bản đồ thời điểm đỉnh
KNEE_BETA = cfg.KNEE_BETA

gt = generate_ground_truth()
al, be, Dg = gt["alpha_grid"], gt["beta_grid"], gt["D_grid"]
params, E_fem = gt["params_flat"], gt["E_fem_flat"]
E_ana = analytical_failure(params[:, 0], params[:, 1], params[:, 2])

# ── Phần 1: chỗ nào biên giải tích sai dấu so với FEM ───────────────────────
disagree = (E_fem > 0) != (E_ana > 0)
knee = params[:, 1] < KNEE_BETA
print("=" * 64)
print("1. Biên giải tích (mức cân bằng) so với FEM")
print("=" * 64)
print(f"  khớp dấu toàn lưới        : {(~disagree).mean()*100:.1f}%")
print(f"  khớp dấu vùng knee β<{KNEE_BETA}  : {(~disagree[knee]).mean()*100:.1f}%"
      f"   ({knee.sum()} điểm)")
print(f"  khớp dấu phần còn lại     : {(~disagree[~knee]).mean()*100:.1f}%"
      f"   ({(~knee).sum()} điểm)")
frac = disagree[knee].sum() / max(disagree.sum(), 1)
print(f"  → {frac*100:.0f}% tổng số điểm sai dấu nằm trong vùng knee, "
      f"trong khi vùng này chỉ chiếm {knee.mean()*100:.0f}% lưới")

# ── Phần 2: u đạt đỉnh lúc nào ───────────────────────────────────────────────
print("\n" + "=" * 64)
print("2. Thời điểm u đạt đỉnh (t*/T).  ~1.0 = cân bằng, <1 = quá độ")
print("=" * 64)

A = np.linspace(*cfg.ALPHA_RANGE, NG)
B = np.linspace(*cfg.BETA_RANGE, NG)
D_show = [float(Dg[1]), float(Dg[len(Dg) // 2]), float(Dg[-2])]

tstar = np.zeros((len(D_show), NG, NG))
Emap = np.zeros_like(tstar)
t0 = time.time()
n = 0
for di, D_val in enumerate(D_show):
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            r = solve_reaction_diffusion(float(a), float(b), D_val, dense=True)
            traj = r["traj"]                       # (nt, nx, nx)
            peak_t = traj.reshape(len(traj), -1).max(axis=1)
            k = int(np.argmax(peak_t))
            tstar[di, i, j] = r["t_eval"][k] / cfg.T_END
            Emap[di, i, j] = r["E"]
            n += 1
    print(f"  D={D_val:.4f}: xong  ({n} lời gọi FEM, {time.time()-t0:.0f}s)")

for di, D_val in enumerate(D_show):
    m_knee = (B[None, :] < KNEE_BETA) & (Emap[di] > 0)
    m_rest = (B[None, :] >= KNEE_BETA) & (Emap[di] > 0)
    tk = tstar[di][m_knee]
    tr = tstar[di][m_rest]
    print(f"  D={D_val:.4f}   t*/T trung bình:  knee={np.mean(tk):.3f} "
          f"(n={len(tk)})   còn lại={np.mean(tr):.3f} (n={len(tr)})")
    print(f"              tỷ lệ đạt đỉnh trước cuối (t*<0.95T): "
          f"knee={np.mean(tk < 0.95)*100:.0f}%   "
          f"còn lại={np.mean(tr < 0.95)*100:.0f}%")

# ── Hình ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
AA, BB = np.meshgrid(A, B, indexing="ij")

for di, D_val in enumerate(D_show):
    k = int(np.argmin(np.abs(Dg - D_val)))
    AG, BG = np.meshgrid(al, be, indexing="ij")
    Et = gt["E_fem"][:, :, k]
    Ea = E_ana.reshape(len(al), len(be), len(Dg))[:, :, k]

    ax = axes[0, di]
    ax.contourf(AG, BG, (np.sign(Et) != np.sign(Ea)).astype(float),
                levels=[0.5, 1.5], colors=["#d62728"], alpha=0.35)
    ax.contour(AG, BG, Et, levels=[0.0], colors="k", linewidths=2.5)
    ax.contour(AG, BG, Ea, levels=[0.0], colors="tab:blue",
               linewidths=2.0, linestyles="--")
    ax.axhline(KNEE_BETA, color="gray", ls=":", lw=1)
    ax.set_title(f"D = {D_val:.4f}   đỏ = giải tích sai dấu", fontsize=10)
    ax.set_xlabel("α"); ax.set_ylabel("β")

    ax = axes[1, di]
    im = ax.pcolormesh(AA, BB, np.where(Emap[di] > 0, tstar[di], np.nan),
                       cmap="viridis", vmin=0, vmax=1, shading="auto")
    ax.contour(AG, BG, Et, levels=[0.0], colors="k", linewidths=2.0)
    ax.axhline(KNEE_BETA, color="w", ls=":", lw=1.5)
    plt.colorbar(im, ax=ax, label="t* / T  (chỉ các ca sập)")
    ax.set_title("thời điểm đạt đỉnh", fontsize=10)
    ax.set_xlabel("α"); ax.set_ylabel("β")

axes[0, 0].legend(handles=[
    plt.Line2D([0], [0], color="k", lw=2.5, label="biên FEM"),
    plt.Line2D([0], [0], color="tab:blue", lw=2, ls="--", label="biên giải tích"),
], loc="upper left", fontsize=8)

fig.suptitle("Vùng knee: biên giải tích sai ở đâu, và u đạt đỉnh lúc nào",
             fontsize=12)
fig.tight_layout()
out = os.path.join(cfg.RESULTS_DIR, "diag_knee.png")
fig.savefig(out, dpi=180, facecolor="white")
print(f"\n✓ {out}")
np.savez_compressed(os.path.join(cfg.RESULTS_DIR, "diag_knee.npz"),
                    tstar=tstar, Emap=Emap, A=A, B=B, D_show=np.array(D_show))