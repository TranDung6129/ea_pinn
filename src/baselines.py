"""
src/baselines.py
----------------
Baseline methods for comparison:

  B2: Vanilla PINN  — static PINN, MSE loss, uniform sampling, no p-space
  B3: BO + FEM      — Bayesian Optimization with GP surrogate + FEM oracle
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, json, time
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from src.fem_oracle import solve_reaction_diffusion, batch_oracle
from src.pinn_model import ParametricPINN, normalise_params, build_model
from src.adaptive_sampling import _sample_bc


# ══════════════════════════════════════════════════════════════════════════════
#  B2: Vanilla PINN
# ══════════════════════════════════════════════════════════════════════════════

class VanillaPINNTrainer:
    """
    Standard PINN for a SINGLE fixed (alpha, beta, D).
    Demonstrates spectral bias failure near the failure boundary.
    Used for ablation A1.
    """

    def __init__(self, alpha: float, beta: float, D: float,
                 seed: int = 42, n_epochs: int = 5000):
        self.alpha   = alpha
        self.beta    = beta
        self.D       = D
        self.seed    = seed
        self.n_epochs= n_epochs
        self.device  = cfg.DEVICE

        torch.manual_seed(seed)
        self.model = build_model(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=cfg.LR_ADAM)

        # Normalise the single parameter vector
        self.p_hat = normalise_params(
            torch.tensor([alpha], device=self.device),
            torch.tensor([beta],  device=self.device),
            torch.tensor([D],     device=self.device),
        ).squeeze(0)

    def train(self) -> dict:
        history = {"loss": [], "l2_error": [], "l_inf_error": []}
        self.model.train()

        for epoch in tqdm(range(self.n_epochs), desc="Vanilla PINN"):
            # Uniform collocation
            xyt = torch.rand(cfg.N_COLLOCATION, 3, device=self.device)
            xyt[:, 2] *= cfg.T_END

            # PDE residual only (vanilla: no event loss, no adaptive sampling)
            from src.loss_functions import loss_pde, loss_bc, loss_ic, sample_bc_points
            l_pde = loss_pde(self.model, xyt, self.p_hat)

            xy_ic = torch.rand(cfg.N_INITIAL, 2, device=self.device)
            l_ic  = loss_ic_fn(self.model, xy_ic, self.p_hat)

            xyt_bc, normals = _sample_bc(cfg.N_BOUNDARY, self.device)
            l_bc  = loss_bc_fn(self.model, xyt_bc, self.p_hat, normals)

            loss  = l_pde + cfg.LAMBDA_BC * l_bc + cfg.LAMBDA_IC * l_ic

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if epoch % 500 == 0:
                l2, linf = self._eval_error()
                history["loss"].append(loss.item())
                history["l2_error"].append(l2)
                history["l_inf_error"].append(linf)
                tqdm.write(f"  ep={epoch:5d}  loss={loss.item():.4e}  "
                           f"L2={l2:.4f}  L∞={linf:.4f}")

        # Final E prediction
        with torch.no_grad():
            xyt_eval = torch.rand(2048, 3, device=self.device)
            xyt_eval[:, 2] *= cfg.T_END
            u_pred = self.model(xyt_eval, self.p_hat)
            E_pred = (u_pred.max() - cfg.U_THRESHOLD).item()

        r = solve_reaction_diffusion(self.alpha, self.beta, self.D)
        E_true = r["E"]

        result = {
            "alpha": self.alpha, "beta": self.beta, "D": self.D,
            "E_pred": E_pred, "E_true": E_true,
            "false_safe": (E_pred < 0 and E_true > 0),
            "history": history, "seed": self.seed,
        }
        out = os.path.join(cfg.RESULTS_DIR,
                           f"vanilla_pinn_a{self.alpha}_b{self.beta}_D{self.D:.4f}_s{self.seed}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=float)
        return result

    def _eval_error(self) -> tuple:
        """L2 and L∞ error against FEM on a coarse evaluation grid."""
        from src.fem_oracle import solve_reaction_diffusion as fem_solve
        r = fem_solve(self.alpha, self.beta, self.D, dense=True)

        nx   = cfg.FEM_NX
        x    = np.linspace(0, 1, nx)
        X, Y = np.meshgrid(x, x)
        t_idx = -1    # final time snapshot

        u_fem = r["traj"][t_idx].ravel()   # (nx²,)
        x_flat = X.ravel(); y_flat = Y.ravel()
        t_val  = r["t_eval"][t_idx]

        xyt = torch.tensor(
            np.column_stack([x_flat, y_flat, np.full(len(x_flat), t_val)]),
            dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(xyt, self.p_hat).cpu().numpy()
        self.model.train()

        u_fem_t = torch.tensor(u_fem, dtype=torch.float32)
        u_pred_t= torch.tensor(u_pred)
        denom   = max(np.abs(u_fem).max(), 1e-8)
        l2   = float(torch.norm(u_pred_t - u_fem_t) / (torch.norm(u_fem_t) + 1e-8))
        linf = float(np.abs(u_pred - u_fem).max()) / denom
        return l2, linf


# Local imports to avoid circular dependency
def loss_ic_fn(model, xy_ic, p_hat):
    from src.loss_functions import loss_ic
    return loss_ic(model, xy_ic, p_hat)

def loss_bc_fn(model, xyt_bc, p_hat, normals):
    from src.loss_functions import loss_bc
    return loss_bc(model, xyt_bc, p_hat, normals)


# ══════════════════════════════════════════════════════════════════════════════
#  B3: Bayesian Optimization + FEM
# ══════════════════════════════════════════════════════════════════════════════

class BayesianOptBaseline:
    """
    Gaussian Process surrogate + Expected Improvement acquisition.
    Uses scikit-optimize (skopt).

    State-of-the-art black-box baseline. Treats the physical system
    as a black box — ignores PDE structure completely.
    """

    def __init__(self, oracle_budget: int = None, seed: int = 42,
                 n_initial_points: int = 10):
        self.oracle_budget    = oracle_budget or cfg.ORACLE_BUDGET
        self.seed             = seed
        self.n_initial_points = n_initial_points

    def run(self) -> dict:
        from skopt import Optimizer
        from skopt.space import Real

        space = [
            Real(*cfg.ALPHA_RANGE, name="alpha"),
            Real(*cfg.BETA_RANGE,  name="beta"),
            Real(*cfg.D_RANGE,     name="D",    prior="log-uniform"),
        ]

        opt = Optimizer(
            dimensions  = space,
            base_estimator = "GP",
            acq_func    = "EI",
            n_initial_points = self.n_initial_points,
            random_state = self.seed,
        )

        history = []
        t_fem_total = 0.0

        pbar = tqdm(range(self.oracle_budget), desc="BO + FEM")
        for i in pbar:
            # Ask for next point
            x_ask = opt.ask()
            alpha, beta, D = x_ask

            # FEM oracle
            t0  = time.time()
            r   = solve_reaction_diffusion(float(alpha), float(beta), float(D))
            dt  = time.time() - t0
            t_fem_total += dt
            E   = r["E"]

            # Tell optimizer (minimize -E to find maximum E = most dangerous)
            opt.tell(x_ask, -E)

            history.append(dict(
                oracle_call=i, alpha=alpha, beta=beta, D=D,
                E_true=float(E), t_fem=dt
            ))
            pbar.set_postfix(E=f"{E:+.3f}", best=f"{max(h['E_true'] for h in history):+.3f}")

        # Build phase diagram: predict E on a grid using the GP model
        result = self._build_phase_diagram(opt, history)
        result["oracle_history"]   = history
        result["t_fem_total"]      = t_fem_total
        result["seed"]             = self.seed
        result["oracle_budget"]    = self.oracle_budget

        out = os.path.join(cfg.RESULTS_DIR, f"bo_fem_seed{self.seed}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\n✓ BO results saved: {out}")
        return result

    def _build_phase_diagram(self, opt, history: list) -> dict:
        """Use GP model to predict E on a coarse grid for phase diagram."""
        n   = 15   # 15³ = 3375 points
        al  = np.linspace(*cfg.ALPHA_RANGE, n)
        be  = np.linspace(*cfg.BETA_RANGE,  n)
        D_  = np.exp(np.linspace(np.log(cfg.D_RANGE[0]),
                                  np.log(cfg.D_RANGE[1]), n))

        ALF, BET, DDD = np.meshgrid(al, be, D_, indexing="ij")
        pts = np.column_stack([ALF.ravel(), BET.ravel(), DDD.ravel()])

        E_pred = -np.array(opt.models[-1].predict(pts))   # GP prediction
        return {
            "grid_alpha": al.tolist(),
            "grid_beta":  be.tolist(),
            "grid_D":     D_.tolist(),
            "E_pred_grid": E_pred.tolist(),
        }
