"""
run_00_smoketest.py — kiểm tra pipeline chạy được, ~2 phút.
Chạy trước mọi thứ khác. Không sinh ground truth đầy đủ.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import freeze_support
freeze_support()

import config as cfg
cfg.ORACLE_BUDGET = 6
cfg.N_SEED_CALLS  = 4
cfg.INIT_STEPS    = 30
cfg.STEPS_PER_CALL = 5
cfg.N_PARAM_BATCH = 8
cfg.N_COLL_PER_P  = 64
cfg.SCAN_GRID     = 8
cfg.FEM_NX        = 20
cfg.FEM_NT        = 30
cfg.FEM_RTOL      = 1e-3

import torch
import numpy as np


def check(label, ok, detail=""):
    print(f"  {'✓' if ok else '✗ FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def main():
    print("\n" + "=" * 55)
    print("SMOKE TEST — FMD-PINN")
    print("=" * 55)

    print("\n[1/5] FEM oracle")
    from src.fem_oracle import solve_reaction_diffusion
    t0 = time.time()
    r = solve_reaction_diffusion(8.0, 1.0, 0.01)
    check("1 lời gọi FEM", r["success"], f"E={r['E']:+.3f}  {time.time()-t0:.2f}s")

    print("\n[2/5] Mô hình + E trên lưới cố định")
    from src.pinn_model import build_model, normalise_params
    from src.loss_functions import failure_functional
    from src.sampling import eval_grid_size
    model = build_model(cfg.DEVICE)
    p = normalise_params(torch.tensor([8.0]), torch.tensor([1.0]),
                         torch.tensor([0.01])).squeeze(0).to(cfg.DEVICE)
    e1 = failure_functional(model, p).item()
    e2 = failure_functional(model, p).item()
    check("E xác định (gọi 2 lần ra cùng số)", abs(e1 - e2) < 1e-6,
          f"E={e1:+.3f}  lưới={eval_grid_size()} điểm")

    P = torch.rand(5, 3, device=cfg.DEVICE)
    check("E theo batch", failure_functional(model, P).shape == torch.Size([5]))

    print("\n[3/5] Gradient dE/dp")
    p_g = P.clone().requires_grad_(True)
    g = torch.autograd.grad(failure_functional(model, p_g, smooth=True).sum(), p_g)[0]
    check("grad_p E qua autodiff", torch.isfinite(g).all(),
          f"|g|={g.norm(dim=1).mean():.4f}")

    print("\n[4/5] Loss vật lý")
    from src.loss_functions import loss_pde, loss_bc, loss_ic
    from src.sampling import collocation, boundary_points, initial_points
    xyt = collocation(128, cfg.DEVICE)
    pp = torch.rand(128, 3, device=cfg.DEVICE)
    xb, nb = boundary_points(32, cfg.DEVICE)
    l = loss_pde(model, xyt, pp)
    check("L_PDE", torch.isfinite(l) and l.item() >= 0, f"{l.item():.3e}")
    check("L_BC", loss_bc(model, xb, pp[:xb.shape[0]], nb).item() >= 0)
    check("L_IC", loss_ic(model, initial_points(32, cfg.DEVICE),
                          pp[:32]).item() >= 0)

    print(f"\n[5/5] Chạy thử {cfg.ORACLE_BUDGET} lời gọi oracle")
    from src.fmd_pinn import FMDPINNTrainer
    t0 = time.time()
    tr = FMDPINNTrainer(seed=42, oracle_budget=cfg.ORACLE_BUDGET)
    FMDPINNTrainer.clear_state(tr.name)
    res = tr.run()
    h = res["oracle_history"]
    check("Đủ số lời gọi", len(h) == cfg.ORACLE_BUDGET, f"{len(h)}")
    check("Có E_true và E_pinn", all("E_true" in r and "E_pinn" in r for r in h))
    check("E_true hợp lý", all(-5 < r["E_true"] < 10 for r in h))

    # Dọn sạch: smoketest chạy với cấu hình tí hon (INIT_STEPS=30, batch 8).
    # Nếu để lại checkpoint, lần chạy thật sau sẽ RESUME từ đó và bỏ qua toàn
    # bộ pha huấn luyện ban đầu, cho ra kết quả vô nghĩa.
    FMDPINNTrainer.clear_state(tr.name)
    print("  ✓  đã dọn checkpoint của smoketest")

    print("\n" + "=" * 55)
    print(f"✓ PASSED   device={cfg.DEVICE}   {time.time()-t0:.0f}s cho vòng chính")
    print("Tiếp theo:  python run_01_ground_truth.py")
    print("=" * 55)


if __name__ == "__main__":
    main()