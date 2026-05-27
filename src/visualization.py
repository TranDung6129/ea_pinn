"""
src/visualization.py
--------------------
All plotting functions for FMD-PINN Benchmark 1.

  plot_phase_diagram_2d   — 2D slice of E(α, β) at fixed D
  plot_oracle_efficiency  — Hausdorff distance vs oracle calls (all methods)
  plot_fsr_curve          — False Safe Rate over oracle calls
  plot_solution_field     — u(x,y,t) comparison PINN vs FEM
  plot_boundary_comparison — ∂C predicted vs ground truth
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg

# Use grayscale-safe color palette (as per paper: "design all figures grayscale-safe")
CMAP_PHASE  = plt.cm.RdBu_r   # collapse=red, stable=blue
CMAP_FIELD  = plt.cm.viridis
COLORS      = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
LINE_STYLES = ["-", "--", "-.", ":", (0,(3,1,1,1))]


def _save(fig, name: str, dpi: int = 150):
    path = os.path.join(cfg.RESULTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  → Saved {path}")


# ── Phase Diagram ─────────────────────────────────────────────────────────────

def plot_phase_diagram_2d(gt: dict,
                           oracle_history_dict: dict = None,
                           D_fixed: float = None,
                           save: bool = True):
    """
    2D phase diagram: E(α, β) at a fixed D slice.

    Parameters
    ----------
    gt                  : ground truth dict from ground_truth.py
    oracle_history_dict : {method_name: oracle_history_list}
    D_fixed             : D value for the slice (default: middle of range)
    """
    D_grid = gt["D_grid"]
    if D_fixed is None:
        D_idx = len(D_grid) // 2
        D_fixed = D_grid[D_idx]
    else:
        D_idx = np.argmin(np.abs(D_grid - D_fixed))

    E_slice = gt["E_fem"][:, :, D_idx]   # (n_alpha, n_beta)
    al = gt["alpha_grid"]
    be = gt["beta_grid"]
    AL, BE = np.meshgrid(al, be, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: FEM ground truth ────────────────────────────────────────────────
    ax = axes[0]
    vmax = max(abs(E_slice.min()), abs(E_slice.max()), 0.1)
    cf = ax.contourf(AL, BE, E_slice, levels=50,
                      cmap=CMAP_PHASE, vmin=-vmax, vmax=vmax)
    ax.contour(AL, BE, E_slice, levels=[0.0],
                colors="k", linewidths=2.5, linestyles="-")
    plt.colorbar(cf, ax=ax, label="E(u) — Failure Functional")
    ax.set_xlabel("α (reaction rate)")
    ax.set_ylabel("β (nonlinear damping)")
    ax.set_title(f"FEM Ground Truth  |  D={D_fixed:.4f}")

    # Overlay analytical boundary
    from src.ground_truth import analytical_failure
    E_anal = analytical_failure(AL, BE, np.full_like(AL, D_fixed))
    ax.contour(AL, BE, E_anal, levels=[0.0],
                colors="gray", linewidths=1.5, linestyles="--",
                alpha=0.7)
    ax.legend(handles=[
        plt.Line2D([0],[0], color="k",    lw=2.5, label="FEM ∂C (true)"),
        plt.Line2D([0],[0], color="gray", lw=1.5, ls="--", label="Analytical approx."),
    ], loc="upper right", fontsize=9)

    # ── Right: Oracle call scatter ───────────────────────────────────────────
    ax = axes[1]
    ax.contourf(AL, BE, E_slice, levels=50,
                 cmap=CMAP_PHASE, vmin=-vmax, vmax=vmax, alpha=0.4)
    ax.contour(AL, BE, E_slice, levels=[0.0],
                colors="k", linewidths=2.5, linestyles="-")

    if oracle_history_dict:
        for (name, history), color, ls in zip(
                oracle_history_dict.items(), COLORS, LINE_STYLES):
            # Filter to this D slice
            pts = [(r["alpha"], r["beta"], r["E_true"]) for r in history
                   if abs(r["D"] - D_fixed) / D_fixed < 0.3]
            if pts:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                es = [p[2] for p in pts]
                sc = ax.scatter(xs, ys, c=["red" if e>0 else "blue" for e in es],
                                s=20, alpha=0.7, label=f"{name} calls", marker="x")
    ax.set_xlabel("α (reaction rate)")
    ax.set_ylabel("β (nonlinear damping)")
    ax.set_title(f"Oracle Calls on Phase Diagram  |  D={D_fixed:.4f}")

    plt.tight_layout()
    if save:
        _save(fig, f"phase_diagram_D{D_fixed:.4f}")
    plt.close()
    return fig


def plot_phase_diagram_3d_slice(gt: dict, save: bool = True):
    """Three 2D slices through the 3D phase diagram."""
    D_grid = gt["D_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, D_idx in zip(axes, [0, len(D_grid)//2, -1]):
        D_val = D_grid[D_idx]
        E_sl  = gt["E_fem"][:, :, D_idx]
        al    = gt["alpha_grid"]
        be    = gt["beta_grid"]
        AL, BE = np.meshgrid(al, be, indexing="ij")
        vmax = max(abs(E_sl.min()), abs(E_sl.max()), 0.1)
        cf = ax.contourf(AL, BE, E_sl, levels=30,
                          cmap=CMAP_PHASE, vmin=-vmax, vmax=vmax)
        ax.contour(AL, BE, E_sl, levels=[0.0], colors="k", linewidths=2)
        plt.colorbar(cf, ax=ax)
        ax.set_xlabel("α"); ax.set_ylabel("β")
        ax.set_title(f"D = {D_val:.4f}")

    plt.suptitle("Phase Diagrams: E(α,β) at 3 D-slices", fontsize=12)
    plt.tight_layout()
    if save:
        _save(fig, "phase_diagram_slices")
    plt.close()


# ── Oracle Efficiency Curves ──────────────────────────────────────────────────

def plot_oracle_efficiency(curves_dict: dict,
                            delta_target: float = 0.1,
                            save: bool = True):
    """
    Hausdorff distance vs oracle call count for all methods.

    curves_dict : {method_name: hausdorff_curve_array}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for (name, curve), color, ls in zip(
            curves_dict.items(), COLORS, LINE_STYLES):
        ax.semilogy(np.arange(1, len(curve)+1), curve,
                     color=color, ls=ls, lw=2, label=name)

    ax.axhline(delta_target, color="k", ls=":", lw=1.5,
                label=f"δ_target = {delta_target}")
    ax.set_xlabel("Oracle Calls (FEM simulations)")
    ax.set_ylabel("Hausdorff Distance δ_H")
    ax.set_title("Oracle Call Efficiency — Failure Boundary Discovery")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        _save(fig, "oracle_efficiency")
    plt.close()
    return fig


# ── FSR Curve ─────────────────────────────────────────────────────────────────

def plot_fsr_curves(fsr_dict: dict, save: bool = True):
    """False Safe Rate vs oracle calls for each method."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for (name, fsr_curve), color, ls in zip(
            fsr_dict.items(), COLORS, LINE_STYLES):
        ax.plot(np.arange(1, len(fsr_curve)+1), np.array(fsr_curve)*100,
                 color=color, ls=ls, lw=2, label=name)
    ax.axhline(cfg.FSR_ALERT_THRESH * 100, color="red", ls=":", lw=1.5,
                label=f"Alert threshold {cfg.FSR_ALERT_THRESH*100:.0f}%")
    ax.set_xlabel("Oracle Calls")
    ax.set_ylabel("False Safe Rate (%)")
    ax.set_title("False Safe Rate — Engineering Safety Metric")
    ax.set_ylim(0, None)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        _save(fig, "fsr_curves")
    plt.close()
    return fig


# ── Solution Field Comparison ─────────────────────────────────────────────────

def plot_solution_field(model, alpha: float, beta: float, D: float,
                         save: bool = True, device: str = None):
    """Side-by-side PINN vs FEM solution at final time."""
    import torch
    from src.pinn_model import normalise_params
    from src.fem_oracle import solve_reaction_diffusion

    device = device or cfg.DEVICE
    fem = solve_reaction_diffusion(alpha, beta, D, dense=True)
    u_fem  = fem["traj"][-1]   # (nx, nx)
    nx     = u_fem.shape[0]

    x = np.linspace(0, 1, nx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    t_val = float(fem["t_eval"][-1])

    p_hat = normalise_params(
        torch.tensor([alpha], device=device),
        torch.tensor([beta],  device=device),
        torch.tensor([D],     device=device),
    ).squeeze(0)
    xyt = torch.tensor(
        np.column_stack([X.ravel(), Y.ravel(),
                         np.full(nx*nx, t_val)]),
        dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        u_pinn = model(xyt, p_hat).cpu().numpy().reshape(nx, nx)
    model.train()

    err = np.abs(u_pinn - u_fem)
    vmax = max(u_fem.max(), u_pinn.max())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = ["FEM (oracle)", "PINN prediction", "|Error|"]
    fields = [u_fem, u_pinn, err]
    cmaps  = [CMAP_FIELD, CMAP_FIELD, "Reds"]

    for ax, title, field, cmap in zip(axes, titles, fields, cmaps):
        vm = vmax if title != "|Error|" else err.max()
        im = ax.pcolormesh(X, Y, field, cmap=cmap,
                            vmin=0 if title=="|Error|" else -vm, vmax=vm)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    E = fem["E"]
    label = "COLLAPSE" if E > 0 else "STABLE"
    fig.suptitle(f"α={alpha:.1f}  β={beta:.1f}  D={D:.4f}  "
                 f"E={E:+.3f}  [{label}]", fontsize=11)
    plt.tight_layout()
    if save:
        _save(fig, f"field_a{alpha:.1f}_b{beta:.1f}_D{D:.4f}")
    plt.close()
    return fig


# ── Summary Table ─────────────────────────────────────────────────────────────

def print_metrics_table(metrics_dict: dict):
    """Pretty-print comparison table of all methods."""
    print("\n" + "="*70)
    print(f"{'Method':<25} {'δ_H':>8} {'N_δ':>6} {'FSR':>7} {'CR':>7} {'W(h)':>7}")
    print("-"*70)
    for name, m in metrics_dict.items():
        dH   = f"{m.get('hausdorff_final') or np.inf:.4f}"
        nd   = str(m.get('n_delta', -1))
        fsr  = f"{m.get('fsr', 0)*100:.1f}%"
        cr   = f"{m.get('coverage_ratio', 0)*100:.1f}%"
        wt   = f"{m.get('wall_clock', {}).get('t_total_h', 0):.2f}"
        print(f"{name:<25} {dH:>8} {nd:>6} {fsr:>7} {cr:>7} {wt:>7}")
    print("="*70)
