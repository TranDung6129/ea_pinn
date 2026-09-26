"""
src/bo_boundary.py
------------------
Baseline thứ hai: BO nhưng ĐI TÌM BIÊN thay vì đi tìm cực đại.

Vì sao cần: BO gốc trong src/baselines.py tối ưu max E, nên nó dồn gần như
toàn bộ ngân sách vào sâu trong miền sập (nhìn log: best=+2.250, các lời gọi
cuối đều quanh +2.24). GP khớp trên tập điểm toàn E dương thì dự đoán dương ở
mọi nơi, tập mức không rỗng, và δ_H ra vô cùng. Đó là một phát hiện thật về
BO chứ không phải lỗi, nhưng nó khiến BO gốc không phải là baseline công bằng
cho bài toán dựng biên.

Biến thể này giữ nguyên mọi thứ của BO (GP + EI + cùng oracle + cùng ngân
sách) và chỉ đổi hàm mục tiêu thành |E|, tức đi tìm E = 0. Khi đó cả hai
phương pháp cùng một mục tiêu, và khác nhau ở chỗ FMD-PINN dùng cấu trúc PDE
còn BO coi hệ là hộp đen.
"""

import numpy as np
import os, json, time
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from src.fem_oracle import solve_reaction_diffusion


class BoundaryBO:
    """GP + EI, mục tiêu min |E| — tìm mặt nghiệm E(p) = 0."""

    def __init__(self, oracle_budget: int = None, seed: int = 42,
                 n_initial_points: int = 10):
        self.oracle_budget = oracle_budget or cfg.ORACLE_BUDGET
        self.seed = seed
        self.n_initial_points = n_initial_points

    def run(self) -> dict:
        from skopt import Optimizer
        from skopt.space import Real

        # Vùng được phép gọi oracle, cùng ràng buộc với FMD-PINN. Vùng cho theo
        # toạ độ chuẩn hoá [0,1]^3 nên phải đổi ngược về đơn vị vật lý; D dùng
        # thang log đúng như normalise_params.
        from src.sampling import region_bounds_np
        lo, hi = region_bounds_np()
        a0, a1 = cfg.ALPHA_RANGE
        b0, b1 = cfg.BETA_RANGE
        d0, d1 = cfg.D_RANGE
        A = (a0 + lo[0] * (a1 - a0), a0 + hi[0] * (a1 - a0))
        B = (b0 + lo[1] * (b1 - b0), b0 + hi[1] * (b1 - b0))
        logd = np.log(d0) + np.array([lo[2], hi[2]]) * (np.log(d1) - np.log(d0))
        D = tuple(np.exp(logd))

        opt = Optimizer(
            dimensions=[Real(*A, name="alpha"),
                        Real(*B, name="beta"),
                        Real(*D, name="D", prior="log-uniform")],
            base_estimator="GP", acq_func="EI",
            n_initial_points=self.n_initial_points,
            random_state=self.seed,
        )
        if not np.allclose(lo, 0) or not np.allclose(hi, 1):
            print(f"  vùng gọi oracle: α∈[{A[0]:.2f},{A[1]:.2f}] "
                  f"β∈[{B[0]:.2f},{B[1]:.2f}] D∈[{D[0]:.4f},{D[1]:.4f}]")

        history = []
        pbar = tqdm(range(self.oracle_budget), desc="BO biên")
        for i in pbar:
            x = opt.ask()
            alpha, beta, D = [float(v) for v in x]
            t0 = time.time()
            E = solve_reaction_diffusion(alpha, beta, D)["E"]
            t_fem = time.time() - t0

            opt.tell(x, abs(E))          # ← khác BO gốc: tối thiểu |E|

            history.append(dict(oracle_call=i, alpha=alpha, beta=beta, D=D,
                                E_true=float(E), t_fem=t_fem))
            pbar.set_postfix(E=f"{E:+.3f}",
                             gan_nhat=f"{min(abs(h['E_true']) for h in history):.3f}")

        result = {"oracle_history": history, "seed": self.seed,
                  "oracle_budget": self.oracle_budget, "objective": "min|E|"}
        out = os.path.join(cfg.RESULTS_DIR, f"bo_boundary_seed{self.seed}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\n✓ {out}")
        return result