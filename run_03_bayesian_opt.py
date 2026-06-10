"""
run_03_bayesian_opt.py
----------------------
STEP 3 — Bayesian Optimization + FEM Baseline (B3)

What it does:
  • GP surrogate with EI acquisition + FEM oracle
  • Same oracle budget as FMD-PINN
  • Records oracle call efficiency curve for comparison
  • Runs with N_SEEDS for statistical validity

Runtime estimate:
  120 FEM calls × ~5s = ~600s ≈ 10 min per seed

Output:
  results/bo_fem_seed*.json
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiprocessing import freeze_support
freeze_support()

import numpy as np
import json
import config as cfg
from src.baselines import BayesianOptBaseline
from src.ground_truth import generate_ground_truth
from src.metrics import compute_all_metrics


def main():
    print("\n" + "="*60)
    print("STEP 3: Bayesian Optimization + FEM Baseline")
    print(f"  oracle_budget={cfg.ORACLE_BUDGET}  n_seeds={cfg.N_SEEDS}")
    print("="*60)

    # Load ground truth
    print("\nLoading ground truth …")
    gt = generate_ground_truth()

    all_metrics = []
    for seed in range(cfg.N_SEEDS):
        print(f"\n── Seed {seed+1}/{cfg.N_SEEDS}")
        t0 = time.time()

        bo = BayesianOptBaseline(
            oracle_budget    = cfg.ORACLE_BUDGET,
            seed             = cfg.BASE_SEED + seed,
            n_initial_points = 10,
        )
        result = bo.run()
        dt = time.time() - t0

        # Compute metrics
        m = compute_all_metrics(
            oracle_history  = result["oracle_history"],
            gt              = gt,
            t_train_gpu_h   = 0.0,    # BO has no GPU training
            delta_target    = 0.1,
        )
        m["seed"] = seed
        m["method"] = "BO+FEM"
        m["wall_clock_actual_s"] = dt
        all_metrics.append(m)

        print(f"  δ_H={m['hausdorff_final']:.4f}  "
              f"N_δ={m['n_delta']}  "
              f"FSR={m['fsr']*100:.1f}%  "
              f"CR={m['coverage_ratio']*100:.1f}%  "
              f"time={dt/60:.1f}min")

    # Summary
    print("\n── BO+FEM Summary across seeds ──")
    for key in ["hausdorff_final", "n_delta", "fsr", "coverage_ratio"]:
        vals = [m.get(key, 0) for m in all_metrics if m.get(key) is not None]
        if vals:
            print(f"  {key:<20}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save aggregate summary
    summary = {
        "method": "BO+FEM",
        "oracle_budget": cfg.ORACLE_BUDGET,
        "metrics_per_seed": all_metrics,
        "hausdorff_mean": float(np.mean([m["hausdorff_final"] for m in all_metrics])),
        "hausdorff_std":  float(np.std( [m["hausdorff_final"] for m in all_metrics])),
        "n_delta_mean":   float(np.mean([m["n_delta"] for m in all_metrics if m["n_delta"] > 0]
                                        or [-1])),
    }
    out = os.path.join(cfg.RESULTS_DIR, "bo_fem_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n✓ BO+FEM summary saved: {out}")
    print(f"\nProceed to: run_04_fmd_pinn.py")


if __name__ == "__main__":
    main()
