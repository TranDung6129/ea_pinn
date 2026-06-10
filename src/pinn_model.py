"""
src/pinn_model.py
-----------------
Parametric PINN for 2D reaction-diffusion.

Architecture:
  • Fourier feature embedding of (x, y, t) → 2·N_FOURIER dim
  • Spatial branch MLP: Fourier → hidden
  • Parameter branch MLP: (α̂, β̂, D̂) → (γ, δ)  [FiLM params]
  • FiLM modulation: h = γ ⊙ h_spatial + δ
  • Output: Linear(FILM_DIM → 1)

Normalisation convention (all in [0,1]):
  α̂ = (α − α_min)/(α_max − α_min)
  β̂ = (β − β_min)/(β_max − β_min)
  D̂ = (log D − log D_min)/(log D_max − log D_min)   ← log-scale
"""

import torch
import torch.nn as nn
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


# ── Parameter normalisation helpers ───────────────────────────────────────────

def normalise_params(alpha: torch.Tensor, beta: torch.Tensor,
                     D: torch.Tensor) -> torch.Tensor:
    """Convert physical params to normalised [0,1]³.  D is log-scaled."""
    a_min, a_max = cfg.ALPHA_RANGE
    b_min, b_max = cfg.BETA_RANGE
    d_min, d_max = cfg.D_RANGE

    a_hat = (alpha - a_min) / (a_max - a_min)
    b_hat = (beta  - b_min) / (b_max - b_min)
    d_hat = (torch.log(D) - np.log(d_min)) / (np.log(d_max) - np.log(d_min))
    return torch.stack([a_hat, b_hat, d_hat], dim=-1)   # (..., 3)


def denormalise_params(p_hat: torch.Tensor) -> tuple:
    """Inverse of normalise_params. Returns (alpha, beta, D)."""
    a_min, a_max = cfg.ALPHA_RANGE
    b_min, b_max = cfg.BETA_RANGE
    d_min, d_max = cfg.D_RANGE

    alpha = p_hat[..., 0] * (a_max - a_min) + a_min
    beta  = p_hat[..., 1] * (b_max - b_min) + b_min
    D     = torch.exp(p_hat[..., 2] * (np.log(d_max) - np.log(d_min)) + np.log(d_min))
    return alpha, beta, D


def clamp_normalised(p_hat: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Hard-clamp to [eps, 1-eps] to keep physical params in valid range."""
    return torch.clamp(p_hat, eps, 1.0 - eps)


# ── Fourier Feature Embedding ──────────────────────────────────────────────────

class FourierEmbedding(nn.Module):
    """
    Random Fourier Features for (x, y, t).
    φ(v) = [sin(2π B v), cos(2π B v)],  B ∈ ℝ^{n_fourier × 3}  fixed.
    Output dim: 2 · n_fourier
    """
    def __init__(self, n_fourier: int = cfg.N_FOURIER,
                 scale: float = cfg.FOURIER_SCALE, seed: int = 0):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        B = torch.randn(n_fourier, 3, generator=rng) * scale
        self.register_buffer("B", B)          # frozen
        self.out_dim = 2 * n_fourier

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """xyt: (..., 3)  →  (..., 2·n_fourier)"""
        B = self.B.to(xyt.device)
        proj = 2.0 * torch.pi * (xyt @ B.T)   # (..., n_fourier)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ── Parametric PINN ────────────────────────────────────────────────────────────

def _mlp(dims: list, activation=nn.Tanh, last_activation=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_activation:
            layers.append(activation())
    return nn.Sequential(*layers)


class ParametricPINN(nn.Module):
    """
    Simplified architecture: concatenate spatial + parameter inputs directly.
    Replaces FiLM which suffered from dead parameter branch.
    Input: (x, y, t, alpha_hat, beta_hat, D_hat) — 6 dims total
    """
    def __init__(self):
        super().__init__()

        # Fourier embedding for spatial-temporal part only
        self.fourier = FourierEmbedding()
        fourier_out  = self.fourier.out_dim   # 2*N_FOURIER

        # Simple MLP: concat(fourier(xyt), p_hat) -> u
        in_dim = fourier_out + 3   # fourier features + 3 params
        hidden = cfg.SPATIAL_HIDDEN
        layers = []
        dims   = [in_dim] + hidden + [1]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, xyt, p_hat):
        N      = xyt.shape[0]
        device = xyt.device
        if p_hat.dim() == 1:
            p_hat = p_hat.unsqueeze(0).expand(N, -1)
        p_hat = p_hat.to(device)
        self.net.to(device)
        h   = self.fourier(xyt)          # (N, fourier_out)
        inp = torch.cat([h, p_hat], dim=-1)  # (N, fourier_out+3)
        u   = self.net(inp).squeeze(-1)  # (N,)
        return u

    def predict_grid(self, alpha, beta, D, nx=32, device=None):
        device = device or next(self.parameters()).device
        self.eval()
        x = torch.linspace(0, 1, nx, device=device)
        t = torch.linspace(0, cfg.T_END, 20, device=device)
        results = []
        p_hat = normalise_params(
            torch.tensor([alpha], device=device),
            torch.tensor([beta],  device=device),
            torch.tensor([D],     device=device),
        ).squeeze(0)
        with torch.no_grad():
            for ti in t:
                XX, YY = torch.meshgrid(x, x, indexing="ij")
                xyt_g  = torch.stack([
                    XX.flatten(), YY.flatten(),
                    ti.expand(nx*nx)
                ], dim=1)
                u = self.forward(xyt_g, p_hat).reshape(nx,nx).cpu().numpy()
                results.append(u)
        return np.array(results)              # (nt, nx, nx)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(device=None) -> ParametricPINN:
    device = device or cfg.DEVICE
    model  = ParametricPINN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PINN] Parameters: {n_params:,}  device={device}")
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_model("cpu")
    xyt   = torch.rand(512, 3)
    p_hat = torch.rand(3)
    u = m(xyt, p_hat)
    print(f"Output shape: {u.shape}  min={u.min():.3f}  max={u.max():.3f}")
