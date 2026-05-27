"""
run_01_ground_truth.py
----------------------
STEP 1 — Generate dense FEM ground truth for Benchmark 1.

What it does:
  • Runs FEM oracle on a 15³ grid of (α, β, D)  → 3375 simulations
  • Saves ground truth to results/ground_truth.npz
  • Plots 3 phase diagram slices

Runtime estimate (6-core CPU):
  15³ = 3375 sims × ~5 s each / 6 parallel = ~47 min

Checkpoint CP1:
  ✓ results/ground_truth.npz exists
  ✓ results/phase_diagram_slices.png shows clear collapse/stable regions
  ✓ Console prints collapse fraction (should be ~30–60%)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg

def main():
    print("\n" + "="*60)
    print("STEP 1: Ground Truth Generation")
    print("="*60)

    # Quick sanity check on FEM oracle
    print("\n[1/3] Sanity check — 3 FEM oracle calls …")
    from src.fem_oracle import solve_reaction_diffusion
    import numpy as np

    test_cases = [
        (10.0, 1.0, 0.01, "should COLLAPSE"),
        (1.5,  4.0, 0.1,  "should STABLE"),
        (3.5,  1.0, 0.02, "near boundary"),
    ]
    for alpha, beta, D, label in test_cases:
        t0 = time.time()
        r  = solve_reaction_diffusion(alpha, beta, D)
        dt = time.time() - t0
        status = "COLLAPSE" if r["E"] > 0 else "STABLE"
        ok     = "✓" if label.split()[-1].upper() == status or "boundary" in label else "✗ UNEXPECTED"
        print(f"  {ok}  α={alpha:.1f} β={beta:.1f} D={D:.3f} → "
              f"E={r['E']:+.3f} [{status}] in {dt:.1f}s")

    # Generate dense ground truth
    print(f"\n[2/3] Running {cfg.GT_GRID}³ = {cfg.GT_GRID**3} FEM simulations …")
    print(f"      (using {cfg.NUM_CPU_WORKERS} parallel workers)")
    print(f"      Cache: {cfg.GT_CACHE}")

    t0 = time.time()
    from src.ground_truth import generate_ground_truth
    gt = generate_ground_truth()
    elapsed = time.time() - t0

    print(f"\n✓ Ground truth complete in {elapsed/60:.1f} min")
    print(f"  Grid size: {gt['n_grid']}³ = {gt['n_grid']**3}")
    print(f"  Collapse fraction: {gt['collapse_frac']:.3f}")
    print(f"  E range: [{gt['E_fem_flat'].min():.3f}, {gt['E_fem_flat'].max():.3f}]")

    # Verify analytical approximation quality
    from src.ground_truth import analytical_failure
    E_anal = analytical_failure(gt["params_flat"][:,0],
                                 gt["params_flat"][:,1],
                                 gt["params_flat"][:,2])
    E_fem  = gt["E_fem_flat"]
    agree  = ((E_fem > 0) == (E_anal > 0)).mean()
    print(f"  Analytical boundary accuracy: {agree:.3f}  (should be > 0.75)")

    # Checkpoint assertion
    assert os.path.exists(cfg.GT_CACHE), "Ground truth cache not saved!"
    assert gt["collapse_frac"] > 0.05, "Too few collapse cases — check parameters!"
    assert gt["collapse_frac"] < 0.95, "Too many collapse cases — check parameters!"
    print("\n✓ CHECKPOINT CP1 PASSED")

    # Plot
    print("\n[3/3] Plotting phase diagram slices …")
    from src.visualization import plot_phase_diagram_3d_slice
    plot_phase_diagram_3d_slice(gt, save=True)
    print("✓ Phase diagram saved")

    # Report boundary points
    from src.ground_truth import get_gt_boundary_points
    bp = get_gt_boundary_points(gt)
    print(f"\n  Boundary points near ∂C: {len(bp)}")
    print(f"  Sample: α={bp[0,0]:.2f}, β={bp[0,1]:.2f}, D={bp[0,2]:.4f}")

    print("\n" + "="*60)
    print("Step 1 complete. Proceed to run_02_vanilla_pinn.py")
    print("="*60)


if __name__ == "__main__":
    main()
