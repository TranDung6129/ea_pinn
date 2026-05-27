"""
run_04_fmd_pinn.py  (patched v2 — PINN-based Hausdorff metric)
------------------
STEP 4 — FMD-PINN (Full Method) + Ablation Studies

Changes vs original:
  1. trainer.load_checkpoint() called before trainer.run()  (resume support)
  2. compute_all_metrics receives model= and ckpt_name_base= for PINN-based δ_H
"""

import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config as cfg
from src.fmd_pinn     import FMDPINNTrainer
from src.ground_truth import generate_ground_truth
from src.metrics      import compute_all_metrics
from src.visualization import (plot_phase_diagram_2d, plot_oracle_efficiency,
                                plot_fsr_curves, print_metrics_table)


ABLATION_VARIANTS = {
    "A2_event_only":    dict(use_event_loss=True,  use_adaptive_samp=False, use_min_max=False),
    "A3_adaptive_only": dict(use_event_loss=False, use_adaptive_samp=True,  use_min_max=False),
    "A4_no_minmax":     dict(use_event_loss=True,  use_adaptive_samp=True,  use_min_max=False),
    "A5_full_fmd":      dict(use_event_loss=True,  use_adaptive_samp=True,  use_min_max=True),
}


def run_variant(name: str, flags: dict, seed: int, gt: dict):
    print(f"\n  [{name}]  seed={seed}  flags={flags}")
    t0 = time.time()

    trainer = FMDPINNTrainer(seed=cfg.BASE_SEED + seed, **flags)
    trainer.load_checkpoint()   # resume from checkpoint if available
    results = trainer.run()

    dt = time.time() - t0

    # Checkpoint base name — must match trainer._save_checkpoint naming exactly
    seed_val = cfg.BASE_SEED + seed
    mm  = flags.get("use_min_max",       True)
    ev  = flags.get("use_event_loss",    True)
    ads = flags.get("use_adaptive_samp", True)
    ckpt_name_base = (f"fmd_pinn_seed{seed_val}"
                      f"_minmax{int(mm)}_ev{int(ev)}_as{int(ads)}")

    m = compute_all_metrics(
        oracle_history = results["oracle_history"],
        gt             = gt,
        t_train_gpu_h  = dt / 3600,
        delta_target   = 0.1,
        model          = trainer.model,      # PINN-based boundary extraction
        ckpt_name_base = ckpt_name_base,     # per-call δ_H curve via checkpoints
    )
    m.update({"variant": name, "seed": seed, "dt_total_s": dt})

    out = os.path.join(cfg.RESULTS_DIR, f"{name}_seed{seed}_metrics.json")
    with open(out, "w") as f:
        json.dump(m, f, indent=2, default=float)
    return m, results, trainer.model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="all",
                        choices=list(ABLATION_VARIANTS.keys()) + ["all"])
    parser.add_argument("--seeds",   type=int, default=cfg.N_SEEDS)
    parser.add_argument("--quick",   action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("STEP 4: FMD-PINN + Ablation Studies")
    print(f"  device={cfg.DEVICE}  batch={cfg.N_COLLOCATION}")
    print("="*60)

    if args.quick:
        print("\n  -- QUICK TEST MODE --")
        cfg.ORACLE_BUDGET = 10; cfg.ADAM_EPOCHS = 60; cfg.LBFGS_STEPS = 5
        cfg.GT_GRID = 4; cfg.N_COLLOCATION = 256; cfg.N_BOUNDARY = 64
        cfg.N_INITIAL = 64; cfg.N_EVENT = 64; cfg.OUTER_STEPS = 3
        cfg.SPATIAL_HIDDEN = [64, 64, 64]; cfg.PARAM_HIDDEN = [32, 32]
        args.seeds = 1; args.variant = "A5_full_fmd"

    print("\nLoading ground truth …")
    gt = generate_ground_truth()

    variants_to_run = (ABLATION_VARIANTS if args.variant == "all"
                       else {args.variant: ABLATION_VARIANTS[args.variant]})

    all_metrics = {}; all_histories = {}; last_models = {}

    for var_name, flags in variants_to_run.items():
        print(f"\n{'='*50}\nVariant: {var_name}\n{'='*50}")
        seed_metrics = []
        for seed in range(args.seeds):
            m, results, model = run_variant(var_name, flags, seed, gt)
            seed_metrics.append(m)
            print(f"  seed={seed}  δ_H={m['hausdorff_final']:.4f}  "
                  f"N_δ={m['n_delta']}  FSR={m['fsr']*100:.1f}%  "
                  f"CR={m['coverage_ratio']*100:.1f}%")
        all_metrics[var_name]   = seed_metrics
        all_histories[var_name] = results["oracle_history"]
        last_models[var_name]   = model

    # Aggregate
    aggregate = {}
    for var_name, seeds in all_metrics.items():
        def agg(key):
            vals = [s[key] for s in seeds if s.get(key) is not None
                    and not (isinstance(s[key], (int,float)) and s[key] == -1)]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)
        aggregate[var_name] = {
            "hausdorff_mean": agg("hausdorff_final")[0],
            "hausdorff_std":  agg("hausdorff_final")[1],
            "n_delta_mean":   agg("n_delta")[0],
            "fsr_mean":       agg("fsr")[0],
            "cr_mean":        agg("coverage_ratio")[0],
        }

    print("\n\n" + "="*70 + "\nABLATION STUDY RESULTS")
    print_metrics_table({k: {
        "hausdorff_final": v["hausdorff_mean"], "n_delta": v["n_delta_mean"],
        "fsr": v["fsr_mean"], "coverage_ratio": v["cr_mean"],
        "wall_clock": {"t_total_h": 0},
    } for k, v in aggregate.items()})

    fmd_dH = aggregate.get("A5_full_fmd", {}).get("hausdorff_mean", np.inf)
    print(f"\n✓ CHECKPOINT CP4:  FMD-PINN δ_H (PINN-based) = {fmd_dH:.4f}")

    # Plots
    hcurves = {v: np.array(s[0].get("hausdorff_curve",[]))
               for v,s in all_metrics.items() if s and s[0].get("hausdorff_curve")}
    if hcurves: plot_oracle_efficiency(hcurves, save=True)

    fsr_c = {v: np.array(s[0].get("fsr_curve",[]))
             for v,s in all_metrics.items() if s and s[0].get("fsr_curve")}
    if fsr_c: plot_fsr_curves(fsr_c, save=True)

    if "A5_full_fmd" in last_models and all_histories.get("A5_full_fmd"):
        plot_phase_diagram_2d(gt=gt,
                              oracle_history_dict={"FMD-PINN": all_histories["A5_full_fmd"]},
                              save=True)

    out = os.path.join(cfg.RESULTS_DIR, "ablation_summary.json")
    with open(out, "w") as f:
        json.dump(aggregate, f, indent=2, default=float)
    print(f"\n✓ Ablation summary saved: {out}\nProceed to: run_05_evaluate.py")


if __name__ == "__main__":
    main()
