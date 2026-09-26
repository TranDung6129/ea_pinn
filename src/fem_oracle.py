"""
src/fem_oracle.py — oracle FEM cho phản ứng khuếch tán 2D.

Hai thay đổi so với bản cũ, cả hai KHÔNG đổi kết quả toán học:

1. Laplacian THƯA thay vì dày.
   Bản cũ dựng ma trận dày (nx*nx) x (nx*nx); với nx=32 là 1024 x 1024 = hơn
   một triệu phần tử, trong khi mỗi hàng chỉ có 5 phần tử khác không. Mỗi lần
   gọi vế phải là một phép nhân ma trận dày ~1e6 flop, và BLAS bung phép nhân
   nhỏ đó ra mọi nhân CPU: chi phí đồng bộ luồng lớn hơn chính phép tính, nên
   càng nhiều nhân càng chậm và máy nào cũng cho cùng thời gian. Ma trận thưa
   giảm còn ~5k flop mỗi lần gọi và chạy một luồng.

2. batch_oracle chạy song song bằng multiprocessing.
   Các lời gọi FEM độc lập hoàn toàn. Bản cũ chạy tuần tự vì lo multiprocessing
   trên Windows; trên Linux thì fork hoạt động bình thường. Mỗi tiến trình con
   ghim về một luồng BLAS để không tranh nhau.

Để kiểm chứng rằng thưa và dày cho cùng một kết quả: python -m src.fem_oracle
"""
import os
import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp
import warnings, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


# ── Laplacian Neumann, dạng thưa ─────────────────────────────────────────────

def _build_laplacian_neumann(nx: int, dx: float, dense: bool = False):
    """
    Sai phân 5 điểm với điều kiện Neumann, phản xạ ở biên (điểm ngoài biên lấy
    bằng điểm đối xứng vào trong). Giống hệt logic bản cũ, chỉ dựng dạng thưa.
    """
    n = nx * nx
    inv = 1.0 / dx ** 2
    rows = np.repeat(np.arange(n), 5)
    cols = np.empty(n * 5, dtype=np.int64)
    vals = np.empty(n * 5, dtype=np.float64)

    k = 0
    for i in range(nx):
        for j in range(nx):
            c = i * nx + j
            jr = j + 1 if j < nx - 1 else j - 1
            jl = j - 1 if j > 0 else j + 1
            iu = i + 1 if i < nx - 1 else i - 1
            idn = i - 1 if i > 0 else i + 1
            cols[k:k + 5] = (c, i * nx + jr, i * nx + jl, iu * nx + j, idn * nx + j)
            vals[k:k + 5] = (-4.0 * inv, inv, inv, inv, inv)
            k += 5

    L = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    L.sum_duplicates()          # biên phản xạ khiến một cột lặp lại: phải cộng
    return L.toarray() if dense else L


_LAP_CACHE = {}


def _lap(nx: int):
    if nx not in _LAP_CACHE:
        _LAP_CACHE[nx] = _build_laplacian_neumann(nx, 1.0 / (nx - 1))
    return _LAP_CACHE[nx]


def _u0(nx: int) -> np.ndarray:
    x = np.linspace(0, 1, nx)
    XX, YY = np.meshgrid(x, x)
    return (0.1 * np.sin(np.pi * XX) * np.sin(np.pi * YY)).ravel()


# ── Một lời gọi oracle ───────────────────────────────────────────────────────

def solve_reaction_diffusion(alpha, beta, D, nx=None, t_end=None, dense=False):
    nx = nx or cfg.FEM_NX
    t_end = t_end or cfg.T_END
    u0 = _u0(nx)
    lap = _lap(nx)

    def rhs(t, u):
        return D * (lap @ u) + alpha * u - beta * u ** 3

    def blowup(t, u):
        return cfg.U_THRESHOLD * 2.5 - u.max()
    blowup.terminal = True
    blowup.direction = -1

    t_eval = np.linspace(0, t_end, cfg.FEM_NT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_ivp(rhs, (0, t_end), u0, method="RK45",
                        t_eval=t_eval, rtol=cfg.FEM_RTOL, atol=cfg.FEM_ATOL,
                        max_step=t_end / 50, events=blowup)

    traj = sol.y.T.reshape(-1, nx, nx)
    max_u = float(traj.max())
    return {"E": max_u - cfg.U_THRESHOLD, "max_u": max_u,
            "success": sol.success, "u_final": traj[-1],
            "traj": traj if dense else None,
            "t_eval": t_eval, "alpha": alpha, "beta": beta, "D": D}


# ── Chạy hàng loạt, song song theo tiến trình ────────────────────────────────

def _limit_threads():
    """Ghim BLAS về một luồng trong tiến trình con, tránh tranh nhân."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(v, "1")


def _worker(row):
    return solve_reaction_diffusion(float(row[0]), float(row[1]), float(row[2]))["E"]


def _init_worker():
    _limit_threads()


def batch_oracle(params, show_progress=True, n_jobs=None):
    """
    Các lời gọi FEM độc lập nhau nên chia thẳng cho nhiều tiến trình.
    n_jobs=1 thì chạy tuần tự trong tiến trình hiện tại (dùng khi gỡ lỗi).
    """
    import time
    params = np.asarray(params, dtype=float)
    N = len(params)
    n_jobs = n_jobs or getattr(cfg, "NUM_CPU_WORKERS", 1)
    n_jobs = max(1, min(int(n_jobs), N))

    _limit_threads()
    t0 = time.time()
    if show_progress:
        sys.stdout.write(f"  {N} mô phỏng FEM trên {n_jobs} tiến trình\n")
        sys.stdout.flush()

    if n_jobs == 1:
        out = []
        for i, row in enumerate(params):
            out.append(_worker(row))
            if show_progress and ((i + 1) % max(1, N // 50) == 0 or i + 1 == N):
                _tick(i + 1, N, t0)
        results = out
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        chunk = max(1, N // (n_jobs * 8))
        results = []
        with ctx.Pool(n_jobs, initializer=_init_worker) as pool:
            last = 0.0
            for i, e in enumerate(pool.imap(_worker, params, chunksize=chunk)):
                results.append(e)
                if show_progress and (time.time() - last > 2.0 or i + 1 == N):
                    _tick(i + 1, N, t0)
                    last = time.time()

    if show_progress:
        el = time.time() - t0
        sys.stdout.write(f"\n  xong {N} mô phỏng trong {el/60:.1f} phút "
                         f"({N/max(el, 1e-9):.1f} sim/s)\n")
        sys.stdout.flush()
    return np.array(results)


def _tick(done, N, t0):
    import time
    el = time.time() - t0
    rate = done / max(el, 1e-9)
    eta = (N - done) / max(rate, 1e-9)
    pct = done / N
    bar = "#" * int(30 * pct) + "-" * (30 - int(30 * pct))
    sys.stdout.write(f"  [{bar}] {done}/{N}  {rate:.1f} sim/s  "
                     f"còn {eta/60:.1f} phút    \r")
    sys.stdout.flush()


# ── Tự kiểm: thưa phải cho cùng kết quả với dày ──────────────────────────────

def verify_sparse_equals_dense(nx=None, n_cases=6):
    """So Laplacian thưa với Laplacian dày, và E của hai đường."""
    nx = nx or cfg.FEM_NX
    dx = 1.0 / (nx - 1)
    Ls = _build_laplacian_neumann(nx, dx)
    Ld = _build_laplacian_neumann(nx, dx, dense=True)
    diff = np.abs(Ls.toarray() - Ld).max()
    print(f"  |L_thưa - L_dày|_max = {diff:.3e}   "
          f"(thưa: {Ls.nnz} phần tử khác 0 trên {nx**2 * nx**2})")

    cases = [(4.0, 4.0, 0.007), (5.0, 2.5, 0.01), (6.0, 1.5, 0.01),
             (10.0, 1.0, 0.01), (2.0, 4.0, 0.1), (14.0, 0.5, 0.005)][:n_cases]
    worst = 0.0
    for a, b, D in cases:
        e_sp = solve_reaction_diffusion(a, b, D)["E"]
        _LAP_CACHE[nx] = Ld                      # ép dùng bản dày
        e_de = solve_reaction_diffusion(a, b, D)["E"]
        _LAP_CACHE[nx] = Ls                      # trả lại bản thưa
        worst = max(worst, abs(e_sp - e_de))
        print(f"  α={a:4.1f} β={b:4.1f} D={D:6.4f}  "
              f"E_thưa={e_sp:+.6f}  E_dày={e_de:+.6f}  lệch {abs(e_sp-e_de):.2e}")
    print(f"\n  lệch E lớn nhất: {worst:.3e}  "
          f"(sai số rời rạc hoá của chính oracle là {cfg.FEM_DISCRETISATION_ERR})")
    return worst


if __name__ == "__main__":
    import time
    print("Kiểm chứng Laplacian thưa so với dày\n")
    verify_sparse_equals_dense()

    print("\nĐo tốc độ một lời gọi")
    t0 = time.time()
    for _ in range(10):
        solve_reaction_diffusion(8.0, 1.0, 0.01)
    print(f"  thưa: {(time.time()-t0)/10*1000:.1f} ms/lời gọi")