"""
diag_seed_variance.py — why do seeds 0,1 win and 2,3,4 lose?

For each A5 seed, extract from its history JSON:
  • whether it resumed or trained fresh (warmup ran or not)
  • early-phase (<50 calls) mean |E_pinn - E_true| in the KNEE region
    (alpha in [3,6], beta < 1.5)  -> proxy for how well it learned the knee
  • final FSR
  • fraction of false-safes in the knee

If knee-region early error correlates with final FSR, warmup/early
learning of the knee is the controlling factor (H1).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config as cfg

SUFFIX = "minmax1_ev1_as1"
TAU = 0.05

def in_knee(a, b):
    return 3.0 <= a <= 6.0 and b < 1.5

print(f"{'seed':<6}{'n_calls':<9}{'knee_err_early':<16}"
      f"{'knee_err_late':<16}{'FSR':<8}{'FS_in_knee':<12}")
print("-" * 70)

for i in range(cfg.N_SEEDS):
    phys = cfg.BASE_SEED + i
    f = os.path.join(cfg.RESULTS_DIR, f"fmd_pinn_seed{phys}_{SUFFIX}.json")
    if not os.path.exists(f):
        print(f"{i:<6}  (missing)")
        continue
    hist = json.load(open(f))["oracle_history"]

    # knee-region prediction error, early vs late
    early_err = [abs(r["E_pinn"] - r["E_true"]) for r in hist[:50]
                 if in_knee(r["alpha"], r["beta"])]
    late_err  = [abs(r["E_pinn"] - r["E_true"]) for r in hist[150:]
                 if in_knee(r["alpha"], r["beta"])]

    # FSR
    coll = [r for r in hist if r.get("E_true", 0) > TAU]
    fs   = [r for r in hist if r.get("E_pinn", -1) < 0 and r.get("E_true", 0) > TAU]
    fsr  = len(fs) / max(1, len(coll))
    fs_knee = sum(1 for r in fs if in_knee(r["alpha"], r["beta"]))
    fs_knee_pct = fs_knee / max(1, len(fs))

    ee = np.mean(early_err) if early_err else float("nan")
    le = np.mean(late_err) if late_err else float("nan")
    print(f"{i:<6}{len(hist):<9}{ee:<16.3f}{le:<16.3f}"
          f"{fsr*100:<8.1f}{fs_knee_pct*100:<12.0f}")

print("\nInterpretation:")
print("  If high early knee_err  →  high FSR : warmup/early knee learning is the cause (H1)")
print("  If FSR high but early knee_err low  : sampling path / outer ascent matters (H2)")