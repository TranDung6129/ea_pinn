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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
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
    fig.savefig(path, dpi=dpi, facecolor="white",
                format='png', metadata=None)
    plt.close(fig)
    print(f"  → Saved {path}")


# ── Phase Diagram ─────────────────────────────────────────────────────────────

def plot_phase_diagram_2d(gt: dict,
                           oracle_history_dict: dict = None,
                           bo_history: list = None,
                           model=None,
                           D_fixed: float = None,
                           save: bool = True):
    """
    4-panel figure for the paper.
    Panel A: FEM ground truth + true boundary ∂C
    Panel B: PINN predicted E(α,β) + predicted boundary
    Panel C: Oracle call locations — FMD-PINN vs BO
    Panel D: Boundary comparison — true vs FMD-PINN vs BO
    """
    import torch
    from src.pinn_model import normalise_params

    D_grid = gt["D_grid"]
    if D_fixed is None:
        D_idx = len(D_grid) // 2
        D_fixed = D_grid[D_idx]
    else:
        D_idx = np.argmin(np.abs(D_grid - D_fixed))

    E_slice = gt["E_fem"][:, :, D_idx]
    al = gt["alpha_grid"]
    be = gt["beta_grid"]
    AL, BE = np.meshgrid(al, be, indexing="ij")
    vmax = max(abs(E_slice.min()), abs(E_slice.max()), 0.1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Failure Manifold Discovery  |  D = {D_fixed:.4f}", fontsize=13)

    # ── Panel A: FEM Ground Truth ─────────────────────────────────────────────
    ax = axes[0, 0]
    cf = ax.contourf(AL, BE, E_slice, levels=50,
                     cmap=CMAP_PHASE, vmin=-vmax, vmax=vmax)
    ax.contour(AL, BE, E_slice, levels=[0.0],
               colors="k", linewidths=2.5, linestyles="-")
    plt.colorbar(cf, ax=ax, label="E(u) — Failure Functional")
    ax.set_xlabel("α (reaction rate)")
    ax.set_ylabel("β (nonlinear damping)")
    ax.set_title("(A) FEM Ground Truth", fontweight="bold")
    ax.text(0.97, 0.97, "True ∂C", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── Panel B: PINN Predicted E(α,β) ───────────────────────────────────────
    ax = axes[0, 1]
    if model is not None:
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            al_flat = torch.tensor(AL.ravel(), dtype=torch.float32, device=device)
            be_flat = torch.tensor(BE.ravel(), dtype=torch.float32, device=device)
            D_flat = torch.full_like(al_flat, D_fixed)
            p_hat = normalise_params(al_flat, be_flat, D_flat)

            n_pts = 512
            xyt = torch.rand(n_pts, 3, device=device)
            xyt[:, 2] *= float(gt.get("T_end", 1.0))

            N = p_hat.shape[0]
            chunk = 100
            E_pinn_flat = []
            for i in range(0, N, chunk):
                pc = p_hat[i:i + chunk]
                nc = pc.shape[0]
                xc = xyt.unsqueeze(0).expand(nc, -1, -1).reshape(-1, 3)
                pp = pc.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, 3)
                u = model(xc, pp).reshape(nc, n_pts)
                E_pinn_flat.append(u.max(dim=1).values.cpu().numpy())
            E_pinn_grid = np.concatenate(E_pinn_flat).reshape(AL.shape) - 1.5
        model.train()

        cf2 = ax.contourf(AL, BE, E_pinn_grid, levels=50,
                          cmap=CMAP_PHASE, vmin=-vmax, vmax=vmax)
        ax.contour(AL, BE, E_slice, levels=[0.0],
                   colors="k", linewidths=2.5, linestyles="-")
        ax.contour(AL, BE, E_pinn_grid, levels=[0.0],
                   colors="lime", linewidths=2.0, linestyles="--")
        plt.colorbar(cf2, ax=ax, label="E_PINN — Predicted")
        ax.legend(handles=[
            plt.Line2D([0], [0], color="k", lw=2.5, label="True ∂C (FEM)"),
            plt.Line2D([0], [0], color="lime", lw=2.0, ls="--", label="Predicted ∂C (PINN)"),
        ], loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Model not loaded", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.contourf(AL, BE, E_slice, levels=50,
                    cmap=CMAP_PHASE, vmin=-vmax, vmax=vmax, alpha=0.3)
    ax.set_xlabel("α (reaction rate)")
    ax.set_ylabel("β (nonlinear damping)")
    ax.set_title("(B) PINN Predicted vs True Boundary", fontweight="bold")

    # ── Panel C: Oracle Call Locations ────────────────────────────────────────
    ax = axes[1, 0]
    ax.contourf(AL, BE, E_slice, levels=50,
                cmap=CMAP_PHASE, vmin=-vmax, vmax=vmax, alpha=0.35)
    ax.contour(AL, BE, E_slice, levels=[0.0],
               colors="k", linewidths=2.5, linestyles="-")

    if oracle_history_dict:
        for name, history in oracle_history_dict.items():
            pts = [(r["alpha"], r["beta"], r["E_true"])
                   for r in history
                   if abs(np.log(r["D"]) - np.log(D_fixed)) < 0.8]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                es = [p[2] for p in pts]
                colors_pts = ["#d62728" if e > 0 else "#1f77b4" for e in es]
                ax.scatter(xs, ys, c=colors_pts, s=35, alpha=0.8,
                           marker="x", linewidths=1.5, zorder=5,
                           label=f"{name} ({len(pts)} calls near slice)")

    if bo_history:
        pts_bo = [(r["alpha"], r["beta"], r["E_true"])
                  for r in bo_history
                  if abs(np.log(r["D"]) - np.log(D_fixed)) < 0.8]
        if pts_bo:
            xs = [p[0] for p in pts_bo]
            ys = [p[1] for p in pts_bo]
            es = [p[2] for p in pts_bo]
            colors_bo = ["#d62728" if e > 0 else "#1f77b4" for e in es]
            ax.scatter(xs, ys, c=colors_bo, s=35, alpha=0.8,
                       marker="o", linewidths=1.5, zorder=4,
                       label=f"BO+FEM ({len(pts_bo)} calls near slice)")

    ax.scatter([], [], c="#d62728", s=35, marker="x", label="Collapse (E>0)")
    ax.scatter([], [], c="#1f77b4", s=35, marker="x", label="Stable (E<0)")
    ax.scatter([], [], c="gray", s=35, marker="o", label="BO+FEM calls")
    ax.legend(loc="upper right", fontsize=7, ncol=1)
    ax.set_xlabel("α (reaction rate)")
    ax.set_ylabel("β (nonlinear damping)")
    ax.set_title("(C) Oracle Call Distribution: FMD-PINN vs BO+FEM", fontweight="bold")

    # ── Panel D: Boundary Comparison ──────────────────────────────────────────
    ax = axes[1, 1]
    ax.contourf(AL, BE, E_slice, levels=50, cmap="Greys", alpha=0.2)
    ax.contour(AL, BE, E_slice, levels=[0.0],
               colors="k", linewidths=3.0, linestyles="-")

    if model is not None:
        ax.contour(AL, BE, E_pinn_grid, levels=[0.0],
                   colors="lime", linewidths=2.0, linestyles="--")

    if bo_history:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        pts_bo_all = [(r["alpha"], r["beta"], 1 if r["E_true"] > 0 else 0)
                      for r in bo_history]
        if len(pts_bo_all) >= 10:
            X_bo = np.array([[p[0], p[1]] for p in pts_bo_all])
            y_bo = np.array([p[2] for p in pts_bo_all])
            try:
                gpc = GaussianProcessClassifier(kernel=ConstantKernel() * RBF())
                gpc.fit(X_bo, y_bo)
                Z_bo = gpc.predict_proba(np.column_stack([AL.ravel(), BE.ravel()]))[:, 1]
                Z_bo = Z_bo.reshape(AL.shape)
                ax.contour(AL, BE, Z_bo, levels=[0.5],
                           colors="orange", linewidths=2.0, linestyles=":")
            except Exception:
                pass

    ax.legend(handles=[
        plt.Line2D([0], [0], color="k", lw=3.0, label="True ∂C (FEM)"),
        plt.Line2D([0], [0], color="lime", lw=2.0, ls="--", label="FMD-PINN ∂C"),
        plt.Line2D([0], [0], color="orange", lw=2.0, ls=":", label="BO+FEM ∂C (GP)"),
    ], loc="upper right", fontsize=9)
    ax.set_xlabel("α (reaction rate)")
    ax.set_ylabel("β (nonlinear damping)")
    ax.set_title("(D) Boundary Accuracy Comparison", fontweight="bold")


    if save:
        _save(fig, "phase_diagram_main")
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
    # plt.tight_layout()
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
    # plt.tight_layout()
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
    # plt.tight_layout()
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
    # plt.tight_layout()
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
