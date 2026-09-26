"""
plot_metric.py — hình từ results/*_metrics.json. Không huấn luyện lại gì.

  python plot_metric.py
  python plot_metric.py --tag cap        # các lần chạy có --tag cap

δ_H vẽ tại đúng các mốc lời gọi có checkpoint (hausdorff_calls), theo chỉ số
cfg.DH_METRIC (mặc định p95). BO+FEM đọc từ bo_fem_summary.json, đã được đo
bằng cùng cách.
"""
import sys, os, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg

DISPLAY = {
    "B1_passive":  ("Thụ động (LHS)",   "tab:gray",   "-."),
    "B2_levelset": ("Tập mức không",    "tab:blue",   "--"),
    "B3_bisect":   ("+ bisection",      "tab:green",  "-"),
    "B4_full":     ("+ ưu tiên knee",   "tab:orange", "-"),
}
DPI = 300
TAG = ""


def load(variant):
    pat = f"{variant}_{TAG}_seed*_metrics.json" if TAG else f"{variant}_seed*_metrics.json"
    return [json.load(open(f))
            for f in sorted(glob.glob(os.path.join(cfg.RESULTS_DIR, pat)))]


def _band(curves):
    curves = [np.asarray(c, float) for c in curves if c is not None and len(c)]
    if not curves:
        return None, None, 0
    L = min(len(c) for c in curves)
    A = np.stack([c[:L] for c in curves])
    A[~np.isfinite(A)] = np.nan
    return np.nanmean(A, 0), np.nanstd(A, 0), L


def plot_hausdorff():
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    target = floor = metric = None

    for var, (label, color, ls) in DISPLAY.items():
        seeds = load(var)
        if not seeds:
            continue
        mean, std, L = _band([m.get("hausdorff_curve") for m in seeds])
        if mean is None:
            continue
        x = np.asarray(seeds[0].get("hausdorff_calls", range(1, L + 1)))[:L]
        ax.plot(x, mean, color=color, ls=ls, lw=2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        target = seeds[0].get("delta_target", target)
        floor = seeds[0].get("delta_floor", floor)
        metric = seeds[0].get("dh_metric", metric)
        nd = [m["n_delta"] for m in seeds if m.get("n_delta", -1) > 0]
        print(f"  {label:<20} n={len(seeds)}  cuối={mean[-1]:.4f}±{std[-1]:.4f}"
              + (f"   N_δ={np.mean(nd):.0f}" if nd else "   N_δ: chưa đạt"))

    for fname, lab, col in [("bo_fem_summary.json", "BO (max E)", "tab:purple"),
                            ("bo_boundary_summary.json", "BO (tìm biên)", "tab:brown")]:
        bo_path = os.path.join(cfg.RESULTS_DIR, fname)
        if not os.path.exists(bo_path):
            continue
        bo = json.load(open(bo_path))
        curves = [m.get("hausdorff_curve")
                  for m in bo.get("metrics_per_seed", []) if m.get("hausdorff_curve")]
        mean, std, L = _band(curves)
        if mean is not None and np.isfinite(mean).any():
            x = np.asarray(bo["metrics_per_seed"][0]["hausdorff_calls"])[:L]
            ax.plot(x, mean, color=col, ls=":", lw=2, label=lab)
            ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.12, lw=0)
            nd = [m["n_delta"] for m in bo["metrics_per_seed"]
                  if m.get("n_delta", -1) > 0]
            print(f"  {lab:<20} n={len(curves)}  cuối={mean[-1]:.4f}±{std[-1]:.4f}"
                  + (f"   N_δ={np.mean(nd):.0f}" if nd else "   N_δ: chưa đạt"))
        else:
            print(f"  {lab:<20} không có biên hữu hạn để vẽ "
                  f"(tập mức không rỗng)")

    if target:
        ax.axhline(target, color="k", ls=":", lw=1, label=f"δ_target = {target:.3f}")
    if floor:
        ax.axhspan(0, floor, color="red", alpha=0.07, lw=0)
        ax.text(0.01, floor, " sàn rời rạc hoá", color="red", fontsize=7,
                va="bottom", transform=ax.get_yaxis_transform())

    ax.set_yscale("log")
    ax.set_xlabel("Số lời gọi oracle (mô phỏng FEM)")
    ax.set_ylabel(f"Sai số biên δ_H [{metric or cfg.DH_METRIC}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(cfg.RESULTS_DIR,
                       f"hausdorff{'_' + TAG if TAG else ''}.png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"✓ {out}")


def plot_safety():
    """FSR dải biên so với báo động giả — hai phía của cùng một đánh đổi."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for var, (label, color, _) in DISPLAY.items():
        seeds = load(var)
        if not seeds:
            continue
        x = [m["far_band"] * 100 for m in seeds if np.isfinite(m.get("far_band", np.nan))]
        y = [m["fsr_band"] * 100 for m in seeds if np.isfinite(m.get("fsr_band", np.nan))]
        if not x:
            continue
        ax.scatter(x, y, color=color, s=60, label=label, zorder=3)
        print(f"  {label:<20} FSR_biên={np.mean(y):.1f}%  "
              f"báo động giả={np.mean(x):.1f}%")

    for fname, lab, col, mk in [
            ("bo_fem_summary.json", "BO (max E)", "tab:purple", "^"),
            ("bo_boundary_summary.json", "BO (tìm biên)", "tab:brown", "v")]:
        bo_path = os.path.join(cfg.RESULTS_DIR, fname)
        if not os.path.exists(bo_path):
            continue
        ms = json.load(open(bo_path)).get("metrics_per_seed", [])
        x = [m["far_band"] * 100 for m in ms if np.isfinite(m.get("far_band", np.nan))]
        y = [m["fsr_band"] * 100 for m in ms if np.isfinite(m.get("fsr_band", np.nan))]
        if x:
            ax.scatter(x, y, color=col, marker=mk, s=60, label=lab, zorder=3)
            print(f"  {lab:<20} FSR_biên={np.mean(y):.1f}%  "
                  f"báo động giả={np.mean(x):.1f}%")

    ax.axhline(5, color="red", ls=":", lw=1)
    ax.axvline(5, color="red", ls=":", lw=1)
    ax.set_xlabel("Báo động giả trong dải biên (%)  — an toàn bị gọi là sập")
    ax.set_ylabel("FSR trong dải biên (%)  — sập bị gọi là an toàn")
    ax.set_title("Góc dưới trái là tốt. Nằm trên một trục nghĩa là\n"
                 "mô hình đang đổi sai lầm này lấy sai lầm kia.", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(cfg.RESULTS_DIR, f"safety{'_' + TAG if TAG else ''}.png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"✓ {out}")


def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    TAG = ap.parse_args().tag
    print(f"[plot] {cfg.RESULTS_DIR}" + (f"  tag={TAG}" if TAG else ""))
    print("\n── Sai số biên ──")
    plot_hausdorff()
    print("\n── An toàn ──")
    plot_safety()


if __name__ == "__main__":
    main()