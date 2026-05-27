"""
src/ground_truth.py
-------------------
Ground truth for Benchmark 1.

Two sources (as per Section 5.1.1):
  1. Dense FEM sweep  — 15³ = 3375 oracle calls on a regular grid
  2. Analytical boundary — Allen-Cahn stability condition:
         ∂C ≈ { α/β ≈ u_threshold² }  (leading-order approximation)
     D contributes via diffusive stabilisation: effective threshold
         α_eff = α − D·(2π²)   (first spatial mode on [0,1]²)
     Full condition: α_eff / β > u_threshold²

Usage:
  python run_01_ground_truth.py
"""

import numpy as np
import os, time
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from src.fem_oracle import solve_reaction_diffusion


# ── Analytical failure boundary ───────────────────────────────────────────────

def analytical_failure(alpha: np.ndarray,
                        beta:  np.ndarray,
                        D:     np.ndarray) -> np.ndarray:
    """
    Returns E_approx = sqrt(alpha_eff / beta) − u_threshold
    where alpha_eff = alpha − D · (2π²)  (first Neumann eigenvalue on [0,1]²)

    Positive  → predicted collapse region
    Negative  → predicted stable region
    Zero locus → analytical ∂C
    """
    k2_first = 2.0 * np.pi**2   # eigenvalue for (1,0) + (0,1) modes on [0,1]²
    alpha_eff = np.maximum(alpha - D * k2_first, 1e-8)
    u_star    = np.sqrt(alpha_eff / beta)         # equilibrium amplitude
    return u_star - cfg.U_THRESHOLD


# ── Dense FEM phase diagram ───────────────────────────────────────────────────

def generate_ground_truth(n_grid: int = None,
                           cache_path: str = None,
                           n_jobs: int = None) -> dict:
    """
    Run FEM on a regular n_grid³ grid of (α, β, D).

    Parameters
    ----------
    n_grid     : grid resolution per axis (default cfg.GT_GRID)
    cache_path : if file exists, load; else compute and save
    n_jobs     : parallel FEM workers

    Returns
    -------
    dict with keys: alpha_grid, beta_grid, D_grid, E_fem, E_analytic, X (meshgrid)
    """
    n_grid     = n_grid     or cfg.GT_GRID
    cache_path = cache_path or cfg.GT_CACHE
    n_jobs     = n_jobs     or cfg.NUM_CPU_WORKERS

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        cached_n = int(data["n_grid"])
        if cached_n == n_grid:
            print(f"[GT] Loading cached ground truth from {cache_path}  (n_grid={cached_n})")
            return {k: data[k] for k in data.files}
        else:
            print(f"[GT] Cache has n_grid={cached_n}, need {n_grid} — regenerating …")
            os.remove(cache_path)

    print(f"[GT] Generating dense FEM phase diagram  ({n_grid}³ = {n_grid**3} calls) …")

    al_vals = np.linspace(*cfg.ALPHA_RANGE, n_grid)
    be_vals = np.linspace(*cfg.BETA_RANGE,  n_grid)
    D_vals  = np.exp(np.linspace(np.log(cfg.D_RANGE[0]),
                                  np.log(cfg.D_RANGE[1]), n_grid))

    ALF, BET, DDD = np.meshgrid(al_vals, be_vals, D_vals, indexing="ij")
    params = np.column_stack([ALF.ravel(), BET.ravel(), DDD.ravel()])  # (N, 3)
    N = len(params)

    # Run in parallel
    from src.fem_oracle import batch_oracle
    t0    = time.time()
    E_fem = batch_oracle(params, n_jobs=n_jobs)
    elapsed = time.time() - t0
    print(f"[GT] {N} FEM calls completed in {elapsed/60:.1f} min")

    E_analytic = analytical_failure(params[:, 0], params[:, 1], params[:, 2])

    E_fem_grid      = E_fem.reshape(n_grid, n_grid, n_grid)
    E_analytic_grid = E_analytic.reshape(n_grid, n_grid, n_grid)

    # Summary stats
    collapse_frac = (E_fem > 0).mean()
    print(f"[GT] Collapse fraction: {collapse_frac:.3f}")

    # Boundary accuracy of analytical approx
    agree = ((E_fem > 0) == (E_analytic > 0)).mean()
    print(f"[GT] Analytical boundary accuracy: {agree:.3f}")

    result = {
        "alpha_grid":     al_vals,
        "beta_grid":      be_vals,
        "D_grid":         D_vals,
        "E_fem_flat":     E_fem,
        "E_fem":          E_fem_grid,
        "E_analytic":     E_analytic_grid,
        "params_flat":    params,
        "elapsed_s":      elapsed,
        "n_grid":         n_grid,
        "collapse_frac":  collapse_frac,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **result)
    print(f"[GT] Saved to {cache_path}")
    return result


def get_gt_boundary_points(gt: dict, n_contour: int = 500) -> np.ndarray:
    """
    Extract parameter vectors on the failure boundary ∂C (E≈0).
    Returns (M, 3) array of [alpha, beta, D] points near ∂C.
    """
    E    = gt["E_fem_flat"]
    params = gt["params_flat"]
    # Points very near the boundary |E| < 0.1
    near  = np.abs(E) < 0.15
    if near.sum() == 0:
        near = np.abs(E) < 0.3
    boundary_pts = params[near]
    # Subsample if too many
    if len(boundary_pts) > n_contour:
        idx = np.random.choice(len(boundary_pts), n_contour, replace=False)
        boundary_pts = boundary_pts[idx]
    return boundary_pts


if __name__ == "__main__":
    gt = generate_ground_truth()
    print(f"\nSample E values:")
    for i in range(5):
        a, b, D = gt["params_flat"][i]
        E = gt["E_fem_flat"][i]
        print(f"  α={a:.2f} β={b:.2f} D={D:.4f}  E={E:+.3f}  "
              f"{'COLLAPSE' if E>0 else 'STABLE'}")
