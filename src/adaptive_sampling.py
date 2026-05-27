"""
src/adaptive_sampling.py
------------------------
Failure-Informed Adaptive Sampling Strategy (FIASS).

Sampling density:
  P(x,y,t;p) ∝ α_base + β_grad·‖∇u_θ‖ + γ_event·exp(E(u_θ)/ε)

Collocation points are CONTINUOUSLY updated during training so that
more points cluster near the failure surface and high-gradient regions.
"""

import torch
import numpy as np
from scipy.spatial import KDTree
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


class AdaptiveSampler:
    """
    Manages collocation points for PDE, BC, and event losses.

    Usage
    -----
    sampler = AdaptiveSampler(device)
    xyt_pde, xyt_bc, normals, xy_ic, xyt_event = sampler.sample(model, p_hat)
    # ... training step ...
    sampler.update(model, p_hat)   # resample every K steps
    """

    def __init__(self, device: str = None,
                 alpha_base: float = 0.4,
                 beta_grad:  float = 0.3,
                 gamma_event:float = 0.3,
                 eps:        float = 0.1,
                 update_freq:int   = 200):
        self.device       = device or cfg.DEVICE
        self.alpha_base   = alpha_base
        self.beta_grad    = beta_grad
        self.gamma_event  = gamma_event
        self.eps          = eps
        self.update_freq  = update_freq

        self._step_count  = 0

        # Initial uniform samples
        self._xyt_pool    = self._uniform_sample(cfg.N_COLLOCATION * 10)
        self._weights     = np.ones(len(self._xyt_pool)) / len(self._xyt_pool)
        self._kdtree      = KDTree(self._xyt_pool.numpy())

    # ── Public API ────────────────────────────────────────────────────────────

    def sample(self, n_pde: int = None, n_bc: int = None,
               n_ic: int = None, n_event: int = None) -> dict:
        """Draw a training batch. Weights are used for PDE points only."""
        n_pde   = n_pde   or cfg.N_COLLOCATION
        n_bc    = n_bc    or cfg.N_BOUNDARY
        n_ic    = n_ic    or cfg.N_INITIAL
        n_event = n_event or cfg.N_EVENT
        dev     = self.device

        # PDE collocation — importance-weighted
        idx = np.random.choice(len(self._xyt_pool),
                               size=n_pde,
                               replace=False,
                               p=self._weights)
        xyt_pde = self._xyt_pool[idx].to(dev)

        # Boundary
        xyt_bc, normals = _sample_bc(n_bc, dev)

        # Initial condition (t=0)
        xy_ic = torch.rand(n_ic, 2, device=dev)

        # Event region: focus near expected failure zone
        xyt_event = self._sample_near_failure(n_event, dev)

        return {
            "xyt_pde":  xyt_pde,
            "xyt_bc":   xyt_bc,
            "normals":  normals,
            "xy_ic":    xy_ic,
            "xyt_event":xyt_event,
        }

    def update(self, model: torch.nn.Module, p_hat: torch.Tensor):
        """
        Recompute sampling weights based on current PINN predictions.
        Call every `update_freq` training steps.
        """
        self._step_count += 1
        if self._step_count % self.update_freq != 0:
            return

        model.eval()
        with torch.no_grad():
            pts = self._xyt_pool.to(self.device)

            # Evaluate u
            N = len(pts)
            if p_hat.dim() == 1:
                p_exp = p_hat.unsqueeze(0).expand(N, -1)
            else:
                p_exp = p_hat

            # Batch to avoid OOM
            batch = 4096
            u_vals = []
            for i in range(0, N, batch):
                u_vals.append(model(pts[i:i+batch], p_exp[i:i+batch]))
            u_vals = torch.cat(u_vals).cpu().numpy()

        # Gradient-magnitude proxy: local variance in neighbourhood
        # (cheap substitute for actual ‖∇u‖ without autodiff over the whole pool)
        grad_proxy = self._local_variance(u_vals)

        # Proximity to failure surface
        E_vals     = u_vals - cfg.U_THRESHOLD
        prox       = np.exp(E_vals / self.eps)
        prox       = np.clip(prox, 0, 1e3)   # prevent explosion in collapse zone

        # Combined weight
        raw = (self.alpha_base
               + self.beta_grad   * grad_proxy / (grad_proxy.max() + 1e-8)
               + self.gamma_event * prox        / (prox.max()       + 1e-8))
        raw /= raw.sum()
        self._weights = raw

        # Inject new points near failure surface (KD-tree guided)
        self._inject_near_failure()
        model.train()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _uniform_sample(self, n: int) -> torch.Tensor:
        return torch.rand(n, 3) * torch.tensor([1.0, 1.0, cfg.T_END])

    def _local_variance(self, u: np.ndarray, k: int = 8) -> np.ndarray:
        """Approx ‖∇u‖ as local variance via k-NN in the pool."""
        pts_np = self._xyt_pool.numpy()
        dists, idxs = self._kdtree.query(pts_np, k=k + 1)
        var = np.var(u[idxs], axis=1)
        return var

    def _sample_near_failure(self, n: int, device: str) -> torch.Tensor:
        """Use high-weight pool regions (near failure) for event points."""
        top_k = min(n * 10, len(self._xyt_pool))
        top_idx = np.argsort(self._weights)[-top_k:]
        chosen  = np.random.choice(top_idx, size=n, replace=n > len(top_idx))
        pts = self._xyt_pool[chosen].to(device)
        # Add small random perturbation to diversify
        pts = pts + torch.randn_like(pts) * 0.02
        pts = torch.clamp(pts, 0.0, 1.0)
        return pts

    def _inject_near_failure(self, n_new: int = 200):
        """Add new candidate points near high-weight regions."""
        top_k   = min(n_new * 5, len(self._xyt_pool))
        top_idx = np.argsort(self._weights)[-top_k:]
        seeds   = self._xyt_pool[top_idx[:n_new]]
        noise   = torch.randn_like(seeds) * 0.05
        new_pts = torch.clamp(seeds + noise, 0.0, 1.0)

        # Evict low-weight points, append new ones
        bot_idx  = np.argsort(self._weights)[:n_new]
        self._xyt_pool[bot_idx] = new_pts
        # Reset weights for new points
        self._weights[bot_idx]  = self._weights.mean()
        self._weights           /= self._weights.sum()
        # Rebuild KD-tree
        self._kdtree = KDTree(self._xyt_pool.numpy())


# ── Boundary sampling (used also from loss_functions) ─────────────────────────

def _sample_bc(n: int, device: str) -> tuple:
    """Sample boundary points on ∂Ω×[0,T] with outward normals."""
    n4 = n // 4
    r  = torch.rand(4, n4, device=device)
    t  = torch.rand(4, n4, device=device) * cfg.T_END
    z  = torch.zeros(n4, device=device)
    o  = torch.ones(n4,  device=device)

    pts = torch.cat([
        torch.stack([r[0], z, t[0]], dim=1),  # y=0
        torch.stack([r[1], o, t[1]], dim=1),  # y=1
        torch.stack([z,    r[2], t[2]], dim=1),  # x=0
        torch.stack([o,    r[3], t[3]], dim=1),  # x=1
    ], dim=0)                                  # (n, 3)

    normals = torch.cat([
        torch.stack([z,  -o,  z],  dim=1),    # y=0  → n=(0,-1,0)
        torch.stack([z,   o,  z],  dim=1),    # y=1  → n=(0,+1,0)
        torch.stack([-o,  z,  z],  dim=1),    # x=0  → n=(-1,0,0)
        torch.stack([o,   z,  z],  dim=1),    # x=1  → n=(+1,0,0)
    ], dim=0)                                  # (n, 3)

    return pts, normals
