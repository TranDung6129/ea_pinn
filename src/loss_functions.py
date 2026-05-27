"""
src/loss_functions.py
---------------------
All loss components for FMD-PINN Benchmark 1.

L_total = L_PDE + λ_BC·L_BC + λ_IC·L_IC + λ_event(t)·L_event

  L_PDE   : PDE residual  ∂u/∂t − D∇²u − αu + βu³ = 0
  L_BC    : Neumann BC   ∂u/∂n = 0  on ∂Ω
  L_IC    : Initial cond  u(x,y,0) = 0.1·sin(πx)·sin(πy)
  L_event : ReLU² barrier  [max(0, E(u))]²
"""

import torch
import torch.nn as nn
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from src.pinn_model import denormalise_params


# ── PDE residual (auto-diff through PINN) ─────────────────────────────────────

def pde_residual(model: nn.Module,
                 xyt: torch.Tensor,
                 p_hat: torch.Tensor) -> torch.Tensor:
    """
    Compute   r = ∂u/∂t − D·(∂²u/∂x² + ∂²u/∂y²) − α·u + β·u³
    using automatic differentiation.

    xyt   : (N, 3)  requires_grad will be set internally
    p_hat : (N, 3) or (3,)
    Returns: (N,) residual
    """
    xyt = xyt.clone().requires_grad_(True)

    N = xyt.shape[0]
    if p_hat.dim() == 1:
        p_hat_exp = p_hat.unsqueeze(0).expand(N, -1)
    else:
        p_hat_exp = p_hat

    u = model(xyt, p_hat_exp)   # (N,)

    # First-order derivatives
    grad_u = torch.autograd.grad(
        u.sum(), xyt, create_graph=True
    )[0]   # (N, 3)  → columns: ∂u/∂x, ∂u/∂y, ∂u/∂t

    u_t = grad_u[:, 2]     # ∂u/∂t
    u_x = grad_u[:, 0]     # ∂u/∂x
    u_y = grad_u[:, 1]     # ∂u/∂y

    # Second-order via another autodiff pass
    u_xx = torch.autograd.grad(u_x.sum(), xyt, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y.sum(), xyt, create_graph=True)[0][:, 1]

    # Recover physical parameters
    alpha, beta, D = denormalise_params(p_hat_exp)   # each (N,)

    residual = u_t - D * (u_xx + u_yy) - alpha * u + beta * u**3
    return residual


def loss_pde(model, xyt, p_hat):
    r = pde_residual(model, xyt, p_hat)
    return (r**2).mean()


# ── Boundary condition loss (Neumann) ─────────────────────────────────────────

def loss_bc(model: nn.Module,
            xyt_bc: torch.Tensor,
            p_hat: torch.Tensor,
            normals: torch.Tensor) -> torch.Tensor:
    """
    Enforce ∂u/∂n = 0 on ∂Ω.
    xyt_bc  : (M, 3)
    normals : (M, 3)  – outward normal (only x,y components nonzero)
    """
    xyt_bc = xyt_bc.clone().requires_grad_(True)
    M = xyt_bc.shape[0]
    if p_hat.dim() == 1:
        p_hat_exp = p_hat.unsqueeze(0).expand(M, -1)
    else:
        p_hat_exp = p_hat

    u = model(xyt_bc, p_hat_exp)
    grad_u = torch.autograd.grad(u.sum(), xyt_bc, create_graph=True)[0]  # (M,3)
    # Normal derivative:  ∂u/∂n = (∇u · n)
    du_dn = (grad_u * normals).sum(dim=-1)   # (M,)
    return (du_dn**2).mean()


def sample_bc_points(n: int, t_end: float, device: str) -> tuple:
    """Sample boundary points and their outward normals on ∂Ω×[0,T]."""
    n4 = n // 4
    t  = torch.rand(n, 1, device=device) * t_end
    z  = torch.zeros(n4, 1, device=device)
    o  = torch.ones(n4,  1, device=device)
    r  = torch.rand(n4,  1, device=device)

    # Bottom y=0, Top y=1, Left x=0, Right x=1
    pts = torch.cat([
        torch.cat([r, z, t[:n4]], dim=1),         # y=0, n=(0,-1,0)
        torch.cat([r, o, t[n4:2*n4]], dim=1),     # y=1, n=(0,+1,0)
        torch.cat([z, r, t[2*n4:3*n4]], dim=1),   # x=0, n=(-1,0,0)
        torch.cat([o, r, t[3*n4:]], dim=1),        # x=1, n=(+1,0,0)
    ], dim=0)

    normals = torch.cat([
        torch.cat([torch.zeros(n4,1,device=device),
                   -torch.ones(n4,1,device=device),
                   torch.zeros(n4,1,device=device)], dim=1),
        torch.cat([torch.zeros(n4,1,device=device),
                    torch.ones(n4,1,device=device),
                   torch.zeros(n4,1,device=device)], dim=1),
        torch.cat([-torch.ones(n4,1,device=device),
                   torch.zeros(n4,2,device=device)], dim=1),
        torch.cat([ torch.ones(n4,1,device=device),
                   torch.zeros(n4,2,device=device)], dim=1),
    ], dim=0)
    return pts, normals


# ── Initial condition loss ────────────────────────────────────────────────────

def loss_ic(model: nn.Module,
            xy_ic: torch.Tensor,
            p_hat: torch.Tensor) -> torch.Tensor:
    """Enforce u(x,y,0) = 0.1·sin(πx)·sin(πy)."""
    M = xy_ic.shape[0]
    # t=0
    xyt_ic = torch.cat([xy_ic,
                         torch.zeros(M, 1, device=xy_ic.device)], dim=1)  # (M,3)
    if p_hat.dim() == 1:
        p_hat_exp = p_hat.unsqueeze(0).expand(M, -1)
    else:
        p_hat_exp = p_hat

    u_pred = model(xyt_ic, p_hat_exp)
    x, y   = xy_ic[:, 0], xy_ic[:, 1]
    u_true = 0.1 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    return ((u_pred - u_true)**2).mean()


# ── Event-Aware loss ──────────────────────────────────────────────────────────

def failure_functional(model: nn.Module,
                       xyt: torch.Tensor,
                       p_hat: torch.Tensor,
                       threshold: float = None) -> torch.Tensor:
    """
    E(u; p) = max_{x,t} u(x,y,t;p) − u_threshold
    Approximated as the MAX over the provided collocation points.
    Differentiable w.r.t. p_hat via the PINN.
    """
    threshold = threshold or cfg.U_THRESHOLD
    N = xyt.shape[0]
    if p_hat.dim() == 1:
        p_hat_exp = p_hat.unsqueeze(0).expand(N, -1)
    else:
        p_hat_exp = p_hat
    u = model(xyt, p_hat_exp)
    return u.max() - threshold


def loss_event(model: nn.Module,
               xyt: torch.Tensor,
               p_hat: torch.Tensor,
               W_penalty: float = 1.0) -> torch.Tensor:
    """
    L_event = mean( [max(0, u_i − threshold)]² ) · W_penalty
    Activates only when the PINN prediction violates the failure criterion.
    """
    N = xyt.shape[0]
    if p_hat.dim() == 1:
        p_hat_exp = p_hat.unsqueeze(0).expand(N, -1)
    else:
        p_hat_exp = p_hat

    u = model(xyt, p_hat_exp)
    violation = torch.clamp(u - cfg.U_THRESHOLD, min=0.0)
    return W_penalty * (violation**2).mean()


# ── Combined loss ─────────────────────────────────────────────────────────────

class AdaptiveLossScheduler:
    """Increases λ_event as E(u) → 0 during training."""

    def __init__(self):
        self.lambda_event = cfg.LAMBDA_EVENT
        self.step_count   = 0

    def step(self, E_current: float):
        """Call after each training step with the current E value."""
        self.step_count += 1
        # Increase weight when near failure boundary
        if abs(E_current) < 0.5:
            scale = 1.0 + 5.0 * np.exp(-abs(E_current) * 4)
            self.lambda_event = min(cfg.LAMBDA_EVENT_MAX,
                                    cfg.LAMBDA_EVENT * scale)
        elif self.step_count % 500 == 0:
            # Gradual growth
            self.lambda_event = min(cfg.LAMBDA_EVENT_MAX,
                                    self.lambda_event * 1.2)

    def get(self) -> float:
        return self.lambda_event


def total_loss(model, xyt_pde, xyt_bc, normals, xy_ic, xyt_event,
               p_hat, lambda_event=1.0):
    """
    Compute full training loss for a given parameter p_hat.

    Returns: dict with 'total', 'pde', 'bc', 'ic', 'event'
    """
    l_pde   = loss_pde(model, xyt_pde, p_hat)
    l_bc    = loss_bc(model, xyt_bc, p_hat, normals)
    l_ic    = loss_ic(model, xy_ic, p_hat)
    l_ev    = loss_event(model, xyt_event, p_hat)
    l_total = l_pde + cfg.LAMBDA_BC * l_bc + cfg.LAMBDA_IC * l_ic + lambda_event * l_ev

    return {
        "total":  l_total,
        "pde":    l_pde.item(),
        "bc":     l_bc.item(),
        "ic":     l_ic.item(),
        "event":  l_ev.item(),
    }
