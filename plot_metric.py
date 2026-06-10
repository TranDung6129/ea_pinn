"""
plot_metrics_v2.py
------------------
Publication-quality metric figures from per-seed metrics JSONs.
Standalone — does NOT retrain anything, only reads results/*.json.

Key differences vs old plots:
  • δ_H curve: prefers PINN-based curve (predicted-boundary accuracy, i.e. BPE)
    over the legacy oracle-distribution curve. Falls back gracefully.
  • All curves: mean ± std band across ALL seeds, not just seed 0.
  • Minimal text overlay; stats printed to console instead.

Usage (Windows or Ubuntu):
    python plot_metrics_v2.py
Outputs:
    results/oracle_efficiency_v2.png
    results/fsr_curves_v2.png
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # avoids Windows font/backend crashes
import matplotlib.pyplot as plt

import config as cfg

# ── Config ────────────────────────────────────────────────────────────────────

DISPLAY = {
    "A2_event_only":    "FMD (event only)",
    "A3_adaptive_only": "FMD (adaptive only)",
    "A4_no_minmax":     "FMD (no min-max)",
    "A5_full_fmd":      "FMD-PINN (full)",
}
STYLE = {
    "FMD (event only)":    dict(color="tab:blue",   ls="-"),
    "FMD (adaptive only)": dict(color="tab:red",    ls="--"),
    "FMD (no min-max)":    dict(color="tab:green",  ls="-."),
    "FMD-PINN (full)":     dict(color="tab:orange", ls="-",  lw=2.2),
    "BO+FEM":              dict(color="tab:purple", ls=":"),
}
# Keys to try for the PINN-based (BPE) Hausdorff curve, in priority order.
PINN_CURVE_KEYS = ["hausdorff_curve_pinn"]   # exact key confirmed in JSON
LEGACY_CURVE_KEY = "hausdorff_curve"
PINN_VARIANTS = {"A5_full_fmd"}
DELTA_TARGET = 0.1
DPI = 300


# ── Loading ───────────────────────────────────────────────────────────────────

def load_seed_metrics(variant: str) -> list:
    files = sorted(glob.glob(
        os.path.join(cfg.RESULTS_DIR, f"{variant}_seed*_metrics.json")))
    out = []
    for f in files:
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def pick_hausdorff_curve(m: dict, variant: str = "") -> tuple:
    """Return (curve, is_pinn_based). Prefers PINN-based key."""
    for k in PINN_CURVE_KEYS:
        if m.get(k):
            return np.asarray(m[k], dtype=float), True
    if m.get(LEGACY_CURVE_KEY):
        is_pinn = variant in PINN_VARIANTS
        return np.asarray(m[LEGACY_CURVE_KEY], dtype=float), is_pinn
    return None, False


def stack_curves(curves: list) -> tuple:
    """Pad/truncate to common length, return (mean, std, n_len). inf → nan."""
    curves = [c for c in curves if c is not None and len(c) > 0]
    if not curves:
        return None, None, 0
    L = min(len(c) for c in curves)
    arr = np.stack([np.asarray(c[:L], dtype=float) for c in curves])
    arr[~np.isfinite(arr)] = np.nan
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr,  axis=0)
    return mean, std, L


def curve_x(n_pts: int, budget: int) -> np.ndarray:
    """Map curve indices to oracle-call axis.
    PINN-based curves are checkpointed every budget/n_pts calls;
    legacy curves are per-call."""
    if n_pts >= budget:                      # per-call legacy curve
        return np.arange(1, n_pts + 1)
    step = budget / n_pts                    # checkpointed curve
    return np.arange(1, n_pts + 1) * step


# ── Figure 1: Oracle efficiency (δ_H vs calls) ───────────────────────────────

def plot_oracle_efficiency_v2(budget: int):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    any_pinn_based = False

    for variant, label in DISPLAY.items():
        seeds = load_seed_metrics(variant)
        if not seeds:
            continue
        curves, pinn_flags = [], []
        for m in seeds:
            c, is_pinn = pick_hausdorff_curve(m, variant)
            if c is not None:
                curves.append(c)
                pinn_flags.append(is_pinn)
        if not curves:
            continue
        any_pinn_based |= any(pinn_flags)
        mean, std, L = stack_curves(curves)
        x = curve_x(L, budget)
        st = STYLE[label]
        ax.plot(x, mean, label=label, **st)
        ax.fill_between(x, mean - std, mean + std,
                        color=st["color"], alpha=0.15, lw=0)
        print(f"  {label:<24} n_seeds={len(curves)}  "
              f"final δ_H = {mean[-1]:.4f} ± {std[-1]:.4f}  "
              f"({'PINN-based' if any(pinn_flags) else 'legacy'})")

    # BO+FEM: flat reference from summary if present
    bo_path = os.path.join(cfg.RESULTS_DIR, "bo_fem_summary.json")
    if os.path.exists(bo_path):
        with open(bo_path) as f:
            bo = json.load(f)
        bo_dh = bo.get("hausdorff_mean") or bo.get("hausdorff_final")
        if bo_dh:
            ax.axhline(bo_dh, **STYLE["BO+FEM"], label="BO+FEM (final)")
            print(f"  {'BO+FEM':<24} final δ_H = {bo_dh:.4f}")

    ax.axhline(DELTA_TARGET, color="k", ls=":", lw=1,
               label=f"δ_target = {DELTA_TARGET}")
    ax.set_yscale("log")
    ax.set_xlabel("Oracle Calls (FEM simulations)")
    ylabel = ("Boundary Prediction Error δ_H (PINN-based)"
              if any_pinn_based else "Hausdorff Distance δ_H")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(cfg.RESULTS_DIR, "oracle_efficiency_v2.png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"✓ Saved {out}")
    if not any_pinn_based:
        print("⚠ No PINN-based curve key found in metrics JSONs — plotted "
              "legacy oracle-distribution curve. Check JSON keys (see below).")


# ── Figure 2: FSR mean±std across seeds ───────────────────────────────────────

def plot_fsr_v2(budget: int):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for variant, label in DISPLAY.items():
        seeds = load_seed_metrics(variant)
        curves = [np.asarray(m["fsr_curve"], dtype=float) * 100
                  for m in seeds if m.get("fsr_curve")]
        if not curves:
            continue
        mean, std, L = stack_curves(curves)
        x = np.arange(1, L + 1)
        st = STYLE[label]
        ax.plot(x, mean, label=label, **st)
        ax.fill_between(x, np.clip(mean - std, 0, None), mean + std,
                        color=st["color"], alpha=0.15, lw=0)
        print(f"  {label:<24} n_seeds={len(curves)}  "
              f"final FSR = {mean[-1]:.1f}% ± {std[-1]:.1f}%")

    ax.axhline(5.0, color="red", ls=":", lw=1, label="Alert threshold 5%")
    ax.set_xlabel("Oracle Calls")
    ax.set_ylabel("False Safe Rate (%)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(cfg.RESULTS_DIR, "fsr_curves_v2.png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"✓ Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    budget = cfg.ORACLE_BUDGET
    print(f"[plot_metrics_v2] results dir = {cfg.RESULTS_DIR}")

    # Diagnostic: show available keys of one A5 metrics file
    sample = glob.glob(os.path.join(cfg.RESULTS_DIR,
                                    "A5_full_fmd_seed0_metrics.json"))
    if sample:
        with open(sample[0]) as f:
            keys = list(json.load(f).keys())
        print(f"[diag] A5 seed0 metric keys: {keys}\n")

    print("── Figure 1: Oracle efficiency ──")
    plot_oracle_efficiency_v2(budget)
    print("\n── Figure 2: FSR curves ──")
    plot_fsr_v2(budget)


if __name__ == "__main__":
    main()