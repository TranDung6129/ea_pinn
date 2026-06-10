"""
src/metrics.py
--------------
All evaluation metrics from Section 6 of the paper.

Primary:
  δ_H    : Hausdorff distance between predicted and true ∂C
  N_δ    : oracle calls needed to reach δ_H < δ_target
  FSR    : False Safe Rate
  CR     : Coverage Ratio
  SI(n)  : Scalability Index (ratio FMD-PINN / BO oracle calls)
  W      : Wall-clock time

Secondary:
  L2, L∞ error vs FEM solution
  Safety Factor Error ΔFS

NOTE on boundary extraction (v2):
  δ_H is now computed from PINN predictions on a dense parameter grid,
  NOT from sparse oracle history points.  This is the scientifically
  correct definition: we ask "how well does the trained surrogate
  represent the failure boundary ∂C?"
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from typing import List, Dict, Optional
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


# ── Hausdorff Distance ────────────────────────────────────────────────────────

def hausdorff_distance(pred_pts, true_pts):
    if len(pred_pts) == 0 or len(true_pts) == 0:
        return np.inf
    pred_pts = np.atleast_2d(pred_pts)
    true_pts = np.atleast_2d(true_pts)
    if pred_pts.shape[1] != 3 or true_pts.shape[1] != 3:
        return np.inf
    if not (np.isfinite(pred_pts).all() and np.isfinite(true_pts).all()):
        return np.inf
    d1 = directed_hausdorff(pred_pts, true_pts)[0]
    d2 = directed_hausdorff(true_pts, pred_pts)[0]
    return max(d1, d2)


def normalise_for_hausdorff(alpha: np.ndarray,
                              beta:  np.ndarray,
                              D:     np.ndarray) -> np.ndarray:
    """Normalise parameter vectors to [0,1]³ for fair distance comparison."""
    a_hat = (alpha - cfg.ALPHA_RANGE[0]) / (cfg.ALPHA_RANGE[1] - cfg.ALPHA_RANGE[0])
    b_hat = (beta  - cfg.BETA_RANGE[0])  / (cfg.BETA_RANGE[1]  - cfg.BETA_RANGE[0])
    d_hat = ((np.log(D) - np.log(cfg.D_RANGE[0])) /
             (np.log(cfg.D_RANGE[1]) - np.log(cfg.D_RANGE[0])))
    return np.column_stack([a_hat, b_hat, d_hat])


# ── PINN-based boundary extraction (v2) ──────────────────────────────────────

def extract_boundary_from_pinn(model,
                                device: str,
                                n_grid: int = 20,
                                n_spatial: int = 256,
                                threshold_band: float = 0.15) -> np.ndarray:
    """
    Extract the predicted failure boundary ∂C from the PINN surrogate
    by evaluating E_pinn on a dense parameter grid.

    Parameters
    ----------
    model         : trained ParametricPINN
    device        : torch device string
    n_grid        : grid resolution per axis  (n_grid³ total evaluations)
    n_spatial     : spatial collocation points used to approximate E = max u − thresh
    threshold_band: |E_pinn| < threshold_band defines the boundary band

    Returns
    -------
    boundary_pts  : (M, 3) normalised parameter array on predicted ∂C
    """
    import torch
    from src.pinn_model import normalise_params

    # Dense parameter grid (log-scale for D)
    alphas = np.linspace(cfg.ALPHA_RANGE[0], cfg.ALPHA_RANGE[1], n_grid)
    betas  = np.linspace(cfg.BETA_RANGE[0],  cfg.BETA_RANGE[1],  n_grid)
    Ds     = np.exp(np.linspace(np.log(cfg.D_RANGE[0]),
                                np.log(cfg.D_RANGE[1]), n_grid))

    AA, BB, DD = np.meshgrid(alphas, betas, Ds, indexing='ij')
    params_phys = np.column_stack([AA.ravel(), BB.ravel(), DD.ravel()])
    # (n_grid³, 3)

    # Fixed spatial-temporal collocation points (same for all parameter evaluations)
    rng = np.random.default_rng(seed=0)
    xyt_np = rng.random((n_spatial, 3)).astype(np.float32)
    xyt_np[:, 2] *= cfg.T_END
    xyt = torch.tensor(xyt_np, device=device)          # (n_spatial, 3)

    model.eval()
    E_vals = np.empty(len(params_phys), dtype=np.float32)
    batch_size = 1024   # conservative 60-70% VRAM usage

    with torch.no_grad():
        for i in range(0, len(params_phys), batch_size):
            batch = params_phys[i : i + batch_size]   # (B, 3)
            n_b   = len(batch)

            # Normalise parameters
            a_t = torch.tensor(batch[:, 0], dtype=torch.float32, device=device)
            b_t = torch.tensor(batch[:, 1], dtype=torch.float32, device=device)
            d_t = torch.tensor(batch[:, 2], dtype=torch.float32, device=device)
            p_hat = normalise_params(a_t, b_t, d_t)   # (B, 3)

            # Expand: for each param point, repeat all spatial points
            # xyt_rep  : (B * n_spatial, 3)
            # p_hat_rep: (B * n_spatial, 3)
            xyt_rep   = xyt.unsqueeze(0).expand(n_b, -1, -1).reshape(n_b * n_spatial, 3)
            p_hat_rep = p_hat.unsqueeze(1).expand(-1, n_spatial, -1).reshape(n_b * n_spatial, 3)

            u = model(xyt_rep, p_hat_rep)           # (B * n_spatial,)
            u = u.reshape(n_b, n_spatial)           # (B, n_spatial)

            E = u.max(dim=1).values.cpu().numpy() - cfg.U_THRESHOLD
            E_vals[i : i + n_b] = E

    model.train()

    # Points near the zero-level set → predicted boundary
    mask = np.abs(E_vals) < threshold_band
    if mask.sum() < 10:
        # Widen band progressively until we have enough points
        for wider in [0.3, 0.5, 1.0]:
            mask = np.abs(E_vals) < wider
            if mask.sum() >= 10:
                break

    boundary_params = params_phys[mask]
    return normalise_for_hausdorff(
        boundary_params[:, 0],
        boundary_params[:, 1],
        boundary_params[:, 2],
    )


# ── Oracle Call Efficiency (PINN-based, v2) ───────────────────────────────────

def oracle_call_efficiency_pinn(ckpt_name_base: str,
                                 oracle_history: List[Dict],
                                 true_boundary: np.ndarray,
                                 delta_target: float = 0.1,
                                 device: str = None,
                                 n_grid: int = 20,
                                 n_spatial: int = 256) -> tuple:
    """
    Compute N_δ and the Hausdorff curve δ_H(t) using PINN boundary extraction.

    For each saved checkpoint (every 10 oracle calls), load the model and
    compute δ_H from the predicted boundary on a dense parameter grid.
    Between checkpoints the value is held constant (step function).

    Parameters
    ----------
    ckpt_name_base : base name without .pt, e.g.
                     'fmd_pinn_seed42_minmax1_ev1_as1'
    oracle_history : list of oracle records (used only for length / call indices)
    true_boundary  : (M, 3) normalised true ∂C from ground truth
    delta_target   : convergence threshold
    device         : torch device
    n_grid         : grid resolution for PINN boundary scan

    Returns
    -------
    n_calls        : int  (-1 if target never reached)
    hausdorff_curve: (n_budget,) float array of δ_H at each oracle call
    """
    import torch
    from src.pinn_model import build_model

    device = device or cfg.DEVICE
    n_budget = len(oracle_history)

    # Find all available checkpoint files
    ckpt_dir = cfg.CKPT_DIR
    ckpt_calls = []
    for call in range(10, n_budget + 1, 10):
        path = os.path.join(ckpt_dir, f"{ckpt_name_base}_call{call}.pt")
        if os.path.exists(path):
            ckpt_calls.append(call)

    # Also check for final checkpoint (no _callN suffix)
    final_ckpt = os.path.join(ckpt_dir, f"{ckpt_name_base}.pt")
    if os.path.exists(final_ckpt) and n_budget not in ckpt_calls:
        ckpt_calls.append(n_budget)

    if not ckpt_calls:
        print(f"  [metrics] No checkpoints found for {ckpt_name_base}, "
              f"falling back to oracle-history boundary.")
        return oracle_call_efficiency_legacy(oracle_history, true_boundary, delta_target)

    print(f"  [metrics] Computing PINN-based δ_H at {len(ckpt_calls)} checkpoints "
          f"(calls: {ckpt_calls[:5]}{'...' if len(ckpt_calls) > 5 else ''})")

    # Evaluate δ_H at each checkpoint
    dH_at_ckpt = {}
    for call in ckpt_calls:
        if call == n_budget and os.path.exists(final_ckpt):
            path = final_ckpt
        else:
            path = os.path.join(ckpt_dir, f"{ckpt_name_base}_call{call}.pt")

        model = build_model(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))

        pred_boundary = extract_boundary_from_pinn(
            model, device, n_grid=n_grid, n_spatial=n_spatial)
        dH = hausdorff_distance(pred_boundary, true_boundary)
        dH_at_ckpt[call] = dH

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Build full curve (step function between checkpoints)
    curve = np.full(n_budget, np.inf)
    prev_dH = np.inf
    ckpt_sorted = sorted(dH_at_ckpt.keys())

    for idx in range(n_budget):
        call = idx + 1
        # Find the most recent checkpoint ≤ call
        relevant = [c for c in ckpt_sorted if c <= call]
        if relevant:
            prev_dH = dH_at_ckpt[relevant[-1]]
        curve[idx] = prev_dH

    reached  = np.where(curve < delta_target)[0]
    n_calls  = int(reached[0]) + 1 if len(reached) > 0 else -1
    return n_calls, curve


# ── Legacy oracle-history boundary (kept for BO baseline) ────────────────────

def extract_boundary_from_results(oracle_history: List[Dict],
                                   E_key: str = "E_true",
                                   threshold_band: float = 0.15) -> np.ndarray:
    """
    Extract boundary from oracle history points where |E| < threshold_band.
    NOTE: Only reliable for BO/grid-search which intentionally samples near ∂C.
          For FMD-PINN use extract_boundary_from_pinn() instead.
    """
    pts = [r for r in oracle_history if abs(r.get(E_key, np.inf)) < threshold_band]
    if len(pts) < 5:
        pts = oracle_history   # fall back
    arr = np.array([[r["alpha"], r["beta"], r["D"]] for r in pts])
    return normalise_for_hausdorff(arr[:, 0], arr[:, 1], arr[:, 2])


def oracle_call_efficiency_legacy(oracle_history: List[Dict],
                                   true_boundary: np.ndarray,
                                   delta_target: float = 0.1) -> tuple:
    """Legacy version: compute δ_H from oracle history (for BO baseline)."""
    curve = []
    for i in range(1, len(oracle_history) + 1):
        pred = extract_boundary_from_results(oracle_history[:i])
        dH   = hausdorff_distance(pred, true_boundary) if len(pred) > 0 else np.inf
        curve.append(dH)
    curve   = np.array(curve)
    reached = np.where(curve < delta_target)[0]
    n_calls = int(reached[0]) + 1 if len(reached) > 0 else -1
    return n_calls, curve


def oracle_call_efficiency(oracle_history: List[Dict],
                            true_boundary: np.ndarray,
                            delta_target: float = 0.1) -> tuple:
    """Alias kept for backward compatibility — calls legacy version."""
    return oracle_call_efficiency_legacy(oracle_history, true_boundary, delta_target)


# ── False Safe Rate ───────────────────────────────────────────────────────────

def false_safe_rate(oracle_history, E_pinn_key="E_pinn", E_true_key="E_true",
                    tau=0.05):
    """
    FSR with safety dead-band tau.

    A false-safe is only counted when the PINN predicts SAFE while the
    true state is a *significant* collapse (E_true > tau). Points within
    |E_true| <= tau lie inside the FEM oracle's own numerical noise band
    around the threshold and are excluded from both numerator and
    denominator — they are not genuine safety failures.

    tau=0.05 matches the FEM solver tolerance near u=U_THRESHOLD.
    """
    true_collapse = [r for r in oracle_history if r.get(E_true_key, 0) > tau]
    if not true_collapse:
        return 0.0
    if not any(E_pinn_key in r for r in oracle_history):
        return float("nan")
    false_safe = [r for r in oracle_history
                  if r.get(E_pinn_key, -1) < 0 and r.get(E_true_key, 0) > tau]
    return len(false_safe) / len(true_collapse)


def false_safe_rate_curve(oracle_history: List[Dict]) -> np.ndarray:
    """Return FSR as a function of oracle call number."""
    return np.array([false_safe_rate(oracle_history[:i])
                     for i in range(1, len(oracle_history) + 1)])


# ── Coverage Ratio ────────────────────────────────────────────────────────────

def coverage_ratio(oracle_history: List[Dict],
                   gt: dict,
                   model=None, device=None) -> float:
    """
    CR = Area(predicted_collapse ∩ true_collapse) / Area(true_collapse)

    When a model is provided, uses PINN predictions on the GT grid.
    Otherwise falls back to a GP interpolant of oracle history.
    """
    params   = gt["params_flat"]
    E_true   = gt["E_fem_flat"]
    true_col = (E_true > 0)

    if model is not None:
        # PINN-based: evaluate model on the GT parameter grid
        import torch
        from src.pinn_model import normalise_params

        device = device or cfg.DEVICE
        model.eval()
        E_pred_list = []
        xyt_np = np.random.default_rng(0).random((256, 3)).astype(np.float32)
        xyt_np[:, 2] *= cfg.T_END
        xyt = torch.tensor(xyt_np, device=device)

        batch_size = 1024  # conservative 60-70% VRAM usage
        with torch.no_grad():
            for i in range(0, len(params), batch_size):
                b = params[i : i + batch_size]
                n_b = len(b)
                a_t = torch.tensor(b[:, 0], dtype=torch.float32, device=device)
                b_t = torch.tensor(b[:, 1], dtype=torch.float32, device=device)
                d_t = torch.tensor(b[:, 2], dtype=torch.float32, device=device)
                p_hat = normalise_params(a_t, b_t, d_t)
                xyt_rep   = xyt.unsqueeze(0).expand(n_b, -1, -1).reshape(n_b * 256, 3)
                p_hat_rep = p_hat.unsqueeze(1).expand(-1, 256, -1).reshape(n_b * 256, 3)
                u = model(xyt_rep, p_hat_rep).reshape(n_b, 256)
                E = u.max(dim=1).values.cpu().numpy() - cfg.U_THRESHOLD
                E_pred_list.append(E)
        model.train()
        E_pred = np.concatenate(E_pred_list)
    else:
        # GP fallback (for BO / when no model available)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        if len(oracle_history) < 5:
            return 0.0
        X = np.array([[r["alpha"], r["beta"], r["D"]] for r in oracle_history])
        y = np.array([r.get("E_pinn", r.get("E_true", 0)) for r in oracle_history])
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True)
        gp.fit(X, y)
        E_pred = gp.predict(params)

    pred_col     = (E_pred > 0)
    intersection = (true_col & pred_col).sum()
    return float(intersection / max(true_col.sum(), 1))


# ── Wall-Clock Time ───────────────────────────────────────────────────────────

def wall_clock_time(oracle_history: List[Dict],
                    t_train_gpu_h: float) -> dict:
    t_fem_total = sum(r.get("t_fem", 0) for r in oracle_history)
    t_total     = t_train_gpu_h * 3600 + t_fem_total
    return {
        "t_train_gpu_s": t_train_gpu_h * 3600,
        "t_fem_total_s": t_fem_total,
        "t_total_s":     t_total,
        "t_total_h":     t_total / 3600,
        "n_oracle":      len(oracle_history),
        "t_fem_mean_s":  t_fem_total / max(len(oracle_history), 1),
    }


# ── Secondary Metrics ─────────────────────────────────────────────────────────

def solution_accuracy(model, oracle_history: List[Dict],
                       device: str = None,
                       n_eval: int = 5) -> dict:
    """L2 and L∞ error of PINN vs FEM at representative parameter configs."""
    import torch
    from src.pinn_model import normalise_params
    from src.fem_oracle import solve_reaction_diffusion

    device = device or cfg.DEVICE
    regions = {"stable": [], "near_failure": [], "collapse": []}

    for r in oracle_history[:n_eval * 3]:
        E = r["E_true"]
        if   E < -0.3:     regions["stable"].append(r)
        elif abs(E) < 0.3: regions["near_failure"].append(r)
        else:              regions["collapse"].append(r)

    results = {}
    for region, pts in regions.items():
        if not pts:
            results[region] = {"l2": None, "linf": None}
            continue
        r = pts[0]
        alpha, beta, D = r["alpha"], r["beta"], r["D"]

        fem  = solve_reaction_diffusion(alpha, beta, D, dense=True)
        traj = fem["traj"]
        nx   = traj.shape[-1]
        x    = np.linspace(0, 1, nx)
        X, Y = np.meshgrid(x, x)
        t_vals = fem["t_eval"]

        p_hat = normalise_params(
            torch.tensor([alpha], device=device),
            torch.tensor([beta],  device=device),
            torch.tensor([D],     device=device),
        ).squeeze(0)

        l2_list, linf_list = [], []
        model.eval()
        for ti_idx in [len(t_vals)//4, len(t_vals)//2, -1]:
            t_val = float(t_vals[ti_idx])
            xyt   = torch.tensor(
                np.column_stack([X.ravel(), Y.ravel(), np.full(nx*nx, t_val)]),
                dtype=torch.float32, device=device)
            with torch.no_grad():
                u_pred = model(xyt, p_hat).cpu().numpy()
            u_fem = traj[ti_idx].ravel()
            denom = max(np.abs(u_fem).max(), 1e-8)
            l2    = np.linalg.norm(u_pred - u_fem) / (np.linalg.norm(u_fem) + 1e-8)
            linf  = np.abs(u_pred - u_fem).max() / denom
            l2_list.append(l2); linf_list.append(linf)
        model.train()
        results[region] = {"l2": float(np.mean(l2_list)),
                           "linf": float(np.mean(linf_list))}
    return results


# ── Summary report ────────────────────────────────────────────────────────────

def compute_all_metrics(oracle_history: List[Dict],
                         gt: dict,
                         t_train_gpu_h: float = 0.0,
                         delta_target: float = 0.1,
                         model=None,
                         ckpt_name_base: Optional[str] = None) -> dict:
    """
    Compute the full metric suite and return as a dict.

    Parameters
    ----------
    oracle_history  : list of oracle call records
    gt              : ground truth dict from generate_ground_truth()
    t_train_gpu_h   : GPU training time in hours
    delta_target    : Hausdorff convergence threshold
    model           : trained ParametricPINN (if provided, use PINN-based δ_H)
    ckpt_name_base  : checkpoint base name for per-call δ_H curve
                      e.g. 'fmd_pinn_seed42_minmax1_ev1_as1'
    """
    # True boundary from dense ground truth
    mask_gt = np.abs(gt["E_fem_flat"]) < 0.15
    true_boundary = normalise_for_hausdorff(
        gt["params_flat"][mask_gt, 0],
        gt["params_flat"][mask_gt, 1],
        gt["params_flat"][mask_gt, 2],
    )

    # ── δ_H and N_δ ──────────────────────────────────────────────────────────
    if ckpt_name_base is not None:
        # Full per-call curve via PINN checkpoints (most accurate)
        n_delta, hausdorff_curve = oracle_call_efficiency_pinn(
            ckpt_name_base, oracle_history, true_boundary,
            delta_target=delta_target, device=cfg.DEVICE,
        )
    elif model is not None:
        # Final δ_H only (no per-call curve) — single evaluation
        pred_boundary = extract_boundary_from_pinn(model, cfg.DEVICE)
        dH_final      = hausdorff_distance(pred_boundary, true_boundary)
        hausdorff_curve = np.full(len(oracle_history), dH_final)
        reached         = np.where(hausdorff_curve < delta_target)[0]
        n_delta         = int(reached[0]) + 1 if len(reached) > 0 else -1
    else:
        # Legacy fallback: oracle-history boundary (BO baseline)
        n_delta, hausdorff_curve = oracle_call_efficiency_legacy(
            oracle_history, true_boundary, delta_target)

    # ── Other metrics ─────────────────────────────────────────────────────────
    fsr       = false_safe_rate(oracle_history)
    cr        = coverage_ratio(oracle_history, gt, model=model, device=cfg.DEVICE)
    wt        = wall_clock_time(oracle_history, t_train_gpu_h)
    fsr_curve = false_safe_rate_curve(oracle_history)

    return {
        "hausdorff_final": float(hausdorff_curve[-1]) if len(hausdorff_curve) > 0 else np.inf,
        "hausdorff_curve": hausdorff_curve.tolist(),
        "hausdorff_curve_pinn": hausdorff_curve.tolist(),  # alias: PINN-based
        "n_delta":         n_delta,
        "fsr":             fsr,
        "fsr_curve":       fsr_curve.tolist(),
        "coverage_ratio":  cr,
        "wall_clock":      wt,
        "delta_target":    delta_target,
    }
