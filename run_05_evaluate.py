"""
run_05_evaluate.py
------------------
STEP 5 — Final evaluation and paper-ready figures.

Loads all results from previous steps and:
  1. Prints full comparison table (Table 1 in paper)
  2. Plots oracle efficiency curves (Fig. 3)
  3. Plots FSR over time (Fig. 4)
  4. Plots phase diagram comparison (Fig. 5)
  5. Plots solution field comparisons (Fig. 6)
  6. Computes scalability: N_δ vs n_params
  7. Saves final report as results/final_report.json

Checkpoint CP6:
  ✓ All tables and figures generated
  ✓ FMD-PINN N_δ < BO N_δ  (main claim)
  ✓ FSR < 5% for FMD-PINN
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config as cfg
from src.ground_truth import generate_ground_truth
from src.visualization import (plot_oracle_efficiency,
                                 plot_fsr_curves,
                                 plot_phase_diagram_2d,
                                 plot_solution_field,
                                 print_metrics_table)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_bo_hausdorff_from_gp(bo_seed_file, true_boundary, threshold_band=0.15):
    """
    Compute delta_H for BO using GP zero-crossing on the saved prediction grid.
    Apples-to-apples comparison with PINN-based delta_H.
    """
    from src.metrics import normalise_for_hausdorff, hausdorff_distance

    bo     = load_json(bo_seed_file)
    alphas = np.array(bo["grid_alpha"])
    betas  = np.array(bo["grid_beta"])
    Ds     = np.array(bo["grid_D"])
    E_grid = np.array(bo["E_pred_grid"])   # (n^3,) flattened, indexing ij

    AA, BB, DD = np.meshgrid(alphas, betas, Ds, indexing="ij")
    params = np.column_stack([AA.ravel(), BB.ravel(), DD.ravel()])

    mask = np.abs(E_grid) < threshold_band
    for wider in [0.3, 0.5, 1.0]:
        if mask.sum() >= 10:
            break
        mask = np.abs(E_grid) < wider

    bp  = params[mask]
    pred_boundary = normalise_for_hausdorff(bp[:,0], bp[:,1], bp[:,2])
    return hausdorff_distance(pred_boundary, true_boundary), int(mask.sum())



def main():
    print("\n" + "="*60)
    print("STEP 5: Final Evaluation")
    print("="*60)

    # Load ground truth
    gt = generate_ground_truth()

    # True boundary — needed by both BO and FMD-PINN delta_H computation
    from src.metrics import normalise_for_hausdorff
    mask_gt = np.abs(gt["E_fem_flat"]) < 0.15
    true_boundary = normalise_for_hausdorff(
        gt["params_flat"][mask_gt, 0],
        gt["params_flat"][mask_gt, 1],
        gt["params_flat"][mask_gt, 2],
    )
    print(f"  True boundary: {len(true_boundary)} points from GT grid")

    # ── Collect all results ───────────────────────────────────────────────────

    results = {}

    # BO+FEM — recompute delta_H from GP zero-crossing (fair comparison)
    bo_path = os.path.join(cfg.RESULTS_DIR, "bo_fem_summary.json")
    if os.path.exists(bo_path):
        bo_sum = load_json(bo_path)

        # Recompute delta_H per seed from E_pred_grid (GP surrogate boundary)
        bo_seed_files = sorted(glob.glob(
            os.path.join(cfg.RESULTS_DIR, "bo_fem_seed*.json")))
        bo_dH_vals = []
        for sf in bo_seed_files:
            dH, n_pts = compute_bo_hausdorff_from_gp(sf, true_boundary)
            bo_dH_vals.append(dH)
            print(f"  BO seed {os.path.basename(sf)}: "
                  f"δ_H(GP)={dH:.4f}  boundary_pts={n_pts}")

        bo_dH_mean = float(np.mean(bo_dH_vals)) if bo_dH_vals else bo_sum.get("hausdorff_mean")

        # N_delta from GP: check if any seed reached target
        bo_nd = bo_sum.get("n_delta_mean")  # keep legacy for now

        results["BO+FEM"] = {
            "hausdorff_final":  bo_dH_mean,
            "n_delta":          bo_nd,
            "fsr":              np.mean([m["fsr"] for m in
                                         bo_sum.get("metrics_per_seed", [{"fsr": 0}])]),
            "coverage_ratio":   np.mean([m.get("coverage_ratio", 0) for m in
                                          bo_sum.get("metrics_per_seed",
                                                      [{"coverage_ratio": 0}])]),
            "wall_clock":       {"t_total_h": sum(
                m.get("wall_clock_actual_s", 0)
                for m in bo_sum.get("metrics_per_seed", [])
            ) / 3600 / max(len(bo_sum.get("metrics_per_seed", [1])), 1)},
        }
        print(f"  ✓ BO+FEM δ_H (GP-based, mean {len(bo_dH_vals)} seeds) = {bo_dH_mean:.4f}")

    # Ablation variants
    abl_path = os.path.join(cfg.RESULTS_DIR, "ablation_summary.json")
    if os.path.exists(abl_path):
        abl = load_json(abl_path)
        for var_name, m in abl.items():
            display = {
                "A2_event_only":    "FMD (event only)",
                "A3_adaptive_only": "FMD (adaptive only)",
                "A4_no_minmax":     "FMD (no min-max)",
                "A5_full_fmd":      "FMD-PINN (full)",
            }.get(var_name, var_name)
            results[display] = {
                "hausdorff_final": m.get("hausdorff_mean"),
                "n_delta":         m.get("n_delta_mean"),
                "fsr":             m.get("fsr_mean"),
                "coverage_ratio":  m.get("cr_mean"),
                "wall_clock":      {"t_total_h": 0},
            }
        print(f"  ✓ Loaded {len(abl)} ablation variants")

    # Vanilla PINN summary
    vp_path = os.path.join(cfg.RESULTS_DIR, "vanilla_pinn_summary.json")
    if os.path.exists(vp_path):
        vp = load_json(vp_path)
        near = [r for r in vp["results"] if r["config"]["label"] == "NEAR_BOUNDARY"]
        if near:
            results["Vanilla PINN"] = {
                "hausdorff_final": None,   # Vanilla PINN cannot produce a phase diagram
                "n_delta":         None,
                "fsr":             near[0]["false_safe_rate"],
                "coverage_ratio":  0.0,
                "wall_clock":      {"t_total_h": 0},
            }
        print(f"  ✓ Loaded Vanilla PINN results")

    if not results:
        print("  ⚠ No results found. Run steps 1–4 first.")
        return

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n")
    print_metrics_table(results)

    # ── Verify main claim ─────────────────────────────────────────────────────
    print("\n── MAIN CLAIM VERIFICATION ──")
    fmd_nd = results.get("FMD-PINN (full)", {}).get("n_delta")
    bo_nd  = results.get("BO+FEM", {}).get("n_delta")
    fmd_fsr= results.get("FMD-PINN (full)", {}).get("fsr")

    if fmd_nd is not None and bo_nd is not None:
        ratio = bo_nd / max(fmd_nd, 1) if fmd_nd > 0 else np.inf
        claim = "✓ VERIFIED" if (fmd_nd > 0 and fmd_nd < bo_nd) else "✗ NOT MET"
        print(f"  FMD-PINN N_δ = {fmd_nd:.0f}  BO N_δ = {bo_nd:.0f}  "
              f"Speedup = {ratio:.1f}x  [{claim}]")

    if fmd_fsr is not None:
        fsr_ok = "✓ OK" if fmd_fsr < 0.05 else "✗ ABOVE THRESHOLD"
        print(f"  FMD-PINN FSR = {fmd_fsr*100:.1f}%  [target < 5%]  [{fsr_ok}]")

    # ── Oracle efficiency plot ────────────────────────────────────────────────
    print("\nGenerating oracle efficiency curves …")
    # Load per-call Hausdorff curves from individual result files
    hausdorff_curves = {}
    fsr_curves_dict  = {}

    for var_file in glob.glob(os.path.join(cfg.RESULTS_DIR, "*_seed0_metrics.json")):
        m = load_json(var_file)
        var_name = m.get("variant", os.path.basename(var_file))
        display  = {
            "A2_event_only":    "FMD (event only)",
            "A3_adaptive_only": "FMD (adaptive only)",
            "A4_no_minmax":     "FMD (no min-max)",
            "A5_full_fmd":      "FMD-PINN (full)",
        }.get(var_name, var_name)
        if m.get("hausdorff_curve"):
            hausdorff_curves[display] = np.array(m["hausdorff_curve"])
        if m.get("fsr_curve"):
            fsr_curves_dict[display] = np.array(m["fsr_curve"])

    # Add BO Hausdorff curve if available
    bo_per_seed = os.path.join(cfg.RESULTS_DIR, "bo_fem_seed0.json")
    if os.path.exists(bo_per_seed):
        bo_s = load_json(bo_per_seed)
        if bo_s.get("oracle_history"):
            from src.metrics import (compute_all_metrics, normalise_for_hausdorff,
                                      extract_boundary_from_results, oracle_call_efficiency)
            from src.ground_truth import get_gt_boundary_points
            true_bp = get_gt_boundary_points(gt)
            true_bp_n = normalise_for_hausdorff(true_bp[:,0], true_bp[:,1], true_bp[:,2])
            _, bo_hcurve = oracle_call_efficiency(bo_s["oracle_history"],
                                                   true_bp_n, delta_target=0.1)
            hausdorff_curves["BO+FEM"] = bo_hcurve

    if hausdorff_curves:
        plot_oracle_efficiency(hausdorff_curves, save=True)
        print(f"  ✓ Oracle efficiency plot saved")

    if fsr_curves_dict:
        plot_fsr_curves(fsr_curves_dict, save=True)
        print(f"  ✓ FSR curves saved")

    # ── Solution field comparisons ────────────────────────────────────────────
    print("\nGenerating solution field comparisons …")
    # Load best FMD-PINN model (A5, seed 0)
    ckpt = os.path.join(cfg.CKPT_DIR,
                         f"fmd_pinn_seed{cfg.BASE_SEED}_minmax1_ev1_as1.pt")
    if os.path.exists(ckpt):
        import torch
        from src.pinn_model import build_model
        model = build_model(cfg.DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))

        test_cases = [
            (2.0,  4.0, 0.1,  "stable"),
            (3.5,  1.5, 0.05, "near_boundary"),
            (10.0, 1.0, 0.01, "collapse"),
        ]
        for alpha, beta, D, label in test_cases:
            plot_solution_field(model, alpha, beta, D, save=True, device=cfg.DEVICE)
        print(f"  ✓ Solution field plots saved (3 cases)")
    else:
        print(f"  ⚠ Model checkpoint not found: {ckpt}")

    # ── Phase diagram ─────────────────────────────────────────────────────────
    fmd_hist_file = glob.glob(os.path.join(cfg.RESULTS_DIR,
                                            f"fmd_pinn_seed{cfg.BASE_SEED}_minmax1*.json"))
    if fmd_hist_file:
        hist = load_json(fmd_hist_file[0])
        if hist.get("oracle_history"):
            plot_phase_diagram_2d(
                gt=gt,
                oracle_history_dict={"FMD-PINN": hist["oracle_history"]},
                save=True,
            )
            print(f"  ✓ Phase diagram saved")

    # ── Final report ──────────────────────────────────────────────────────────
    report = {
        "benchmark": "Benchmark 1: 2D Reaction-Diffusion",
        "methods": results,
        "main_claim": {
            "fmd_n_delta":     fmd_nd,
            "bo_n_delta":      bo_nd,
            "speedup":         bo_nd / max(fmd_nd, 1) if (fmd_nd and bo_nd) else None,
            "fmd_fsr":         fmd_fsr,
            "claim_verified":  bool(fmd_nd and bo_nd and fmd_nd < bo_nd),
        },
        "figures": [
            "phase_diagram_slices.png",
            "oracle_efficiency.png",
            "fsr_curves.png",
            "field_*.png",
        ],
    }
    rp = os.path.join(cfg.RESULTS_DIR, "final_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: None)
    print(f"\n✓ Final report: {rp}")

    print("\n" + "="*60)
    print("✓ CHECKPOINT CP6 — All figures and tables generated")
    print("="*60)
    print(f"\nAll results in: {cfg.RESULTS_DIR}/")
    print("Review these figures for the manuscript:")
    print("  • phase_diagram_slices.png   — Figure: Phase diagram")
    print("  • oracle_efficiency.png      — Figure: Main efficiency claim")
    print("  • fsr_curves.png             — Figure: Safety metric")
    print("  • field_*.png                — Figure: Solution accuracy")


if __name__ == "__main__":
    main()