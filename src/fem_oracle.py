"""
src/fem_oracle.py  --  FEM oracle for 2D reaction-diffusion
"""
import numpy as np
from scipy.integrate import solve_ivp
import warnings, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


def _build_laplacian_neumann(nx, dx):
    n = nx * nx
    L = np.zeros((n, n))
    for i in range(nx):
        for j in range(nx):
            k = i * nx + j
            L[k, k] = -4.0 / dx**2
            jr = j + 1 if j < nx - 1 else j - 1
            L[k, i * nx + jr] += 1.0 / dx**2
            jl = j - 1 if j > 0 else j + 1
            L[k, i * nx + jl] += 1.0 / dx**2
            iu = i + 1 if i < nx - 1 else i - 1
            L[k, iu * nx + j] += 1.0 / dx**2
            id_ = i - 1 if i > 0 else i + 1
            L[k, id_ * nx + j] += 1.0 / dx**2
    return L


_NX   = cfg.FEM_NX
_DX   = 1.0 / (_NX - 1)
_LAP  = _build_laplacian_neumann(_NX, _DX)
_XX, _YY = np.meshgrid(np.linspace(0,1,_NX), np.linspace(0,1,_NX))
_U0_FLAT = (0.1 * np.sin(np.pi * _XX) * np.sin(np.pi * _YY)).ravel()


def solve_reaction_diffusion(alpha, beta, D, nx=None, t_end=None, dense=False):
    nx    = nx    or cfg.FEM_NX
    t_end = t_end or cfg.T_END
    u0    = _U0_FLAT.copy() if nx == _NX else (
        0.1 * np.sin(np.pi * np.linspace(0,1,nx)[np.newaxis,:]) *
              np.sin(np.pi * np.linspace(0,1,nx)[:,np.newaxis])).ravel()
    lap   = _LAP if nx == _NX else _build_laplacian_neumann(nx, 1.0/(nx-1))

    def rhs(t, u):
        return D * (lap @ u) + alpha * u - beta * u**3

    def blowup(t, u):
        return cfg.U_THRESHOLD * 2.5 - u.max()
    blowup.terminal  = True
    blowup.direction = -1

    t_eval = np.linspace(0, t_end, cfg.FEM_NT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_ivp(rhs, (0, t_end), u0, method="RK45",
                        t_eval=t_eval, rtol=cfg.FEM_RTOL, atol=cfg.FEM_ATOL,
                        max_step=t_end/50, events=blowup)

    traj  = sol.y.T.reshape(-1, nx, nx)
    max_u = float(traj.max())
    return {"E": max_u - cfg.U_THRESHOLD, "max_u": max_u,
            "success": sol.success, "u_final": traj[-1],
            "traj": traj if dense else None,
            "t_eval": t_eval, "alpha": alpha, "beta": beta, "D": D}


def _oracle_worker(row):
    """Single FEM oracle call."""
    alpha, beta, D = float(row[0]), float(row[1]), float(row[2])
    return solve_reaction_diffusion(alpha, beta, D)["E"]


def batch_oracle(params, show_progress=True):
    """Sequential execution — Windows-safe, no multiprocessing."""
    import time

    N = len(params)

    if show_progress:
        sys.stdout.write(f"  Running {N} FEM simulations (sequential)...\n")
        sys.stdout.flush()

    results = []
    t0  = time.time()
    rep = max(1, N // 40)

    for i, row in enumerate(params):
        results.append(_oracle_worker(row))
        done = i + 1
        if show_progress and (done % rep == 0 or done == N):
            el   = time.time() - t0
            rate = done / max(el, 0.001)
            eta  = (N - done) / max(rate, 0.001)
            pct  = done / N
            bar  = "#" * int(25 * pct) + "-" * (25 - int(25 * pct))
            sys.stdout.write(f"  [{bar}] {done}/{N} {rate:.1f}sim/s ETA {eta/60:.0f}min    \r")
            sys.stdout.flush()

    if show_progress:
        el = time.time() - t0
        sys.stdout.write(f"\n  Done {N} sims in {el/60:.1f}min ({N/max(el, 0.001):.1f}sim/s)\n")
        sys.stdout.flush()

    return np.array(results)