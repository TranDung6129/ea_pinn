"""
run_02_vanilla_pinn.py
----------------------
STEP 2 — Vanilla PINN Baseline (Ablation A1)

What it does:
  • Trains a static PINN on 3 representative (α, β, D) configs:
      - Clearly stable
      - Near-boundary (hardest case)
      - Clearly collapse
  • Documents spectral bias failure near failure boundary
  • Runs with N_SEEDS random seeds for statistical validity

Checkpoint CP3:
  ✓ False-stable prediction documented at near-boundary case
  ✓ L2 error spikes near E(u) > 0 region
  ✓ results/vanilla_pinn_*.json files created
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiprocessing import freeze_support
freeze_support()

import numpy as np
import config as cfg
from src.baselines import VanillaPINNTrainer
from src.fem_oracle import solve_reaction_diffusion


# Test configurations: stable | near-boundary | collapse
TEST_CONFIGS = [
    dict(alpha=2.0,  beta=4.0, D=0.1,  label="STABLE"),
    dict(alpha=3.5,  beta=1.5, D=0.05, label="NEAR_BOUNDARY"),
    dict(alpha=10.0, beta=1.0, D=0.01, label="COLLAPSE"),
]


def main():
    print("\n" + "="*60)
    print("STEP 2: Vanilla PINN Baseline")
    print(f"  device={cfg.DEVICE}  n_seeds={cfg.N_SEEDS}")
    print("="*60)

    all_results = []
    false_safe_cases = 0
    total_near = 0

    for cfg_case in TEST_CONFIGS:
        alpha = cfg_case["alpha"]
        beta  = cfg_case["beta"]
        D     = cfg_case["D"]
        label = cfg_case["label"]

        # Ground truth first
        print(f"\n── Config: α={alpha} β={beta} D={D}  [{label}]")
        r_fem = solve_reaction_diffusion(alpha, beta, D)
        print(f"   FEM truth: E={r_fem['E']:+.3f}  max_u={r_fem['max_u']:.3f}")

        seed_results = []
        for seed in range(cfg.N_SEEDS):
            print(f"  [seed {seed+1}/{cfg.N_SEEDS}]", end=" ", flush=True)
            t0 = time.time()
            trainer = VanillaPINNTrainer(alpha, beta, D,
                                          seed=cfg.BASE_SEED + seed,
                                          n_epochs=cfg.ADAM_EPOCHS)
            result = trainer.train()
            dt = time.time() - t0

            seed_results.append(result)
            status = "FALSE_SAFE!" if result["false_safe"] else "OK"
            print(f"E_pred={result['E_pred']:+.4f}  E_true={result['E_true']:+.4f}  "
                  f"[{status}]  {dt:.0f}s")

            if label == "NEAR_BOUNDARY":
                total_near += 1
                if result["false_safe"]:
                    false_safe_cases += 1

        # Summary for this config
        E_preds = [r["E_pred"] for r in seed_results]
        E_true  = seed_results[0]["E_true"]
        print(f"  Summary: E_pred = {np.mean(E_preds):.4f} ± {np.std(E_preds):.4f}  "
              f"E_true = {E_true:+.4f}")

        all_results.append({
            "config": cfg_case,
            "E_true": E_true,
            "E_pred_mean": float(np.mean(E_preds)),
            "E_pred_std":  float(np.std(E_preds)),
            "false_safe_rate": sum(r["false_safe"] for r in seed_results) / cfg.N_SEEDS,
        })

    # Checkpoint CP3
    print("\n" + "="*60)
    print("CHECKPOINT CP3 — Vanilla PINN Failure Documentation")
    print("="*60)

    near_fsr = false_safe_cases / max(total_near, 1)
    print(f"  Near-boundary False Safe Rate: {near_fsr:.2f}  "
          f"({false_safe_cases}/{total_near} seeds)")
    print(f"  (Vanilla PINN is expected to fail — high FSR is a desired result here)")

    # Print summary table
    print(f"\n{'Config':<15} {'E_true':>8} {'E_pred':>10} {'FSR':>6}")
    print("-"*45)
    for r in all_results:
        name  = r["config"]["label"]
        print(f"  {name:<13} {r['E_true']:>+8.3f} "
              f"{r['E_pred_mean']:>+8.3f}±{r['E_pred_std']:.3f} "
              f"{r['false_safe_rate']*100:>5.0f}%")

    # Save summary
    import json
    summary = {"method": "vanilla_pinn", "results": all_results,
                "near_fsr": near_fsr}
    out = os.path.join(cfg.RESULTS_DIR, "vanilla_pinn_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n✓ CHECKPOINT CP3 {'PASSED' if near_fsr > 0.2 else 'NOTE: FSR lower than expected'}")
    print(f"  (Near-boundary FSR={near_fsr:.2f} — high = Vanilla PINN fails as expected)")
    print(f"\nProceed to: run_03_bayesian_opt.py")


if __name__ == "__main__":
    main()
