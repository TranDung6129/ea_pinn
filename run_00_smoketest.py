"""
run_00_smoketest.py
-------------------
Nhanh nhất có thể: ~2–3 phút trên CPU, không sinh GT đầy đủ.
Kiểm tra toàn bộ pipeline chạy được, không crash.

Chạy cái này TRƯỚC KHI chạy run_01...run_05.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiprocessing import freeze_support
freeze_support()

# ── Override config TRƯỚC khi import bất kỳ module nào ───────────────────────
import config as cfg
cfg.ORACLE_BUDGET  = 4        # chỉ 4 FEM calls cho FMD-PINN
cfg.ADAM_EPOCHS    = 50
cfg.LBFGS_STEPS    = 3
cfg.GT_GRID        = 4        # 4³=64 FEM calls cho GT
cfg.N_COLLOCATION  = 128
cfg.N_BOUNDARY     = 32
cfg.N_INITIAL      = 32
cfg.N_EVENT        = 32
cfg.OUTER_STEPS    = 2
cfg.SPATIAL_HIDDEN = [64, 64, 64]
cfg.PARAM_HIDDEN   = [32, 32]
cfg.FILM_DIM       = 64
cfg.FEM_NX         = 20       # lưới thô hơn → mỗi FEM call ~0.1s
cfg.FEM_NT         = 30
cfg.FEM_RTOL       = 1e-3
cfg.N_SEEDS        = 1

import numpy as np

def check(label, ok, detail=""):
    status = "✓" if ok else "✗ FAIL"
    print(f"  {status}  {label}", f"({detail})" if detail else "")
    if not ok:
        sys.exit(1)

def main():
    print("\n" + "="*55)
    print("SMOKE TEST — FMD-PINN Benchmark 1")
    print("="*55)

    # ── 1. FEM oracle ─────────────────────────────────────────────────────────────
    print("\n[1/6] FEM oracle …")
    from src.fem_oracle import solve_reaction_diffusion, batch_oracle
    t0 = time.time()
    r  = solve_reaction_diffusion(8.0, 1.0, 0.01)
    t_single = time.time() - t0
    check("Single FEM call", r["success"], f"E={r['E']:.3f}  t={t_single:.2f}s")

    t0 = time.time()
    E_batch = batch_oracle(np.array([[8.,1.,0.01],[1.5,4.,0.1],[3.5,1.5,0.05]]))
    check("Batch FEM (3 calls, sequential)", len(E_batch)==3,
          f"E={[round(float(e),2) for e in E_batch]}  t={time.time()-t0:.1f}s")

    # ── 2. PINN model ─────────────────────────────────────────────────────────────
    print("\n[2/6] PINN model …")
    import torch
    from src.pinn_model import build_model, normalise_params, denormalise_params
    model = build_model(cfg.DEVICE)
    xyt   = torch.rand(32, 3)
    p_hat = normalise_params(torch.tensor([8.]), torch.tensor([1.]), torch.tensor([0.01])).squeeze(0)
    out   = model(xyt, p_hat)
    check("Forward pass", out.shape == torch.Size([32]), f"shape={out.shape}")
    a, b, D = denormalise_params(p_hat.unsqueeze(0))
    check("Denormalise params", True, f"α={a.item():.2f} β={b.item():.2f} D={D.item():.5f}")

    # ── 3. Loss functions ─────────────────────────────────────────────────────────
    print("\n[3/6] Loss functions …")
    from src.loss_functions import loss_pde, loss_ic, loss_event, failure_functional
    xyt_g = xyt.clone().requires_grad_(True)
    l_pde = loss_pde(model, xyt_g, p_hat)
    check("L_PDE autodiff", l_pde.item() >= 0, f"val={l_pde.item():.4e}")

    xy_ic = torch.rand(32, 2)
    l_ic  = loss_ic(model, xy_ic, p_hat)
    check("L_IC", l_ic.item() >= 0, f"val={l_ic.item():.4e}")

    xyt_ev = torch.rand(32, 3)
    E_pinn = failure_functional(model, xyt_ev, p_hat)
    check("Failure functional E(u)", True, f"E={E_pinn.item():.3f}")

    # Check ∂E/∂p̂ gradient flows
    p_test = p_hat.clone().requires_grad_(True)
    E_test = failure_functional(model, xyt_ev, p_test)
    E_test.backward()
    check("∂E/∂p̂ via autodiff", p_test.grad is not None,
          f"grad={[round(float(g),4) for g in p_test.grad]}")

    # ── 4. Adaptive sampler ───────────────────────────────────────────────────────
    print("\n[4/6] Adaptive sampler …")
    from src.adaptive_sampling import AdaptiveSampler
    sampler = AdaptiveSampler("cpu")
    batch   = sampler.sample(n_pde=128, n_bc=32, n_ic=32, n_event=32)
    check("Sampler batch", all(k in batch for k in
          ["xyt_pde","xyt_bc","normals","xy_ic","xyt_event"]),
          f"keys={list(batch.keys())}")

    # ── 5. Mini ground truth (64 FEM calls) ───────────────────────────────────────
    print(f"\n[5/6] Mini ground truth (4³={4**3} FEM calls) …")
    import os
    if os.path.exists(cfg.GT_CACHE):
        os.remove(cfg.GT_CACHE)
    from src.ground_truth import generate_ground_truth
    t0 = time.time()
    gt = generate_ground_truth(n_grid=4)
    t_gt = time.time() - t0
    check("GT generation", gt["collapse_frac"] > 0.05,
          f"collapse={gt['collapse_frac']:.2f}  t={t_gt:.0f}s")
    check("GT E range sane",
          float(gt["E_fem_flat"].min()) < 0 < float(gt["E_fem_flat"].max()),
          f"E∈[{gt['E_fem_flat'].min():.2f},{gt['E_fem_flat'].max():.2f}]")

    # ── 6. FMD-PINN mini run (4 oracle calls) ─────────────────────────────────────
    print(f"\n[6/6] FMD-PINN mini run (4 oracle calls) …")
    t0 = time.time()
    from src.fmd_pinn import FMDPINNTrainer
    trainer = FMDPINNTrainer(seed=42, use_min_max=True,
                              use_event_loss=True, use_adaptive_samp=True,
                              oracle_budget=4)
    results = trainer.run()
    t_fmd = time.time() - t0
    h = results["oracle_history"]
    check("Oracle calls completed", len(h) == 4, f"calls={len(h)}")
    check("E values recorded",
          all("E_true" in r and "E_pinn" in r for r in h), "")
    check("Results saved",
          os.path.exists(os.path.join(cfg.CKPT_DIR,
              "fmd_pinn_seed42_minmax1_ev1_as1.pt")), "checkpoint exists")

    E_true_vals = [r["E_true"] for r in h]
    check("FEM E values reasonable",
          min(E_true_vals) > -5 and max(E_true_vals) < 10,
          f"E_true={[round(e,2) for e in E_true_vals]}")

    # ── Summary ────────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("✓ ALL SMOKE TESTS PASSED")
    print(f"  GPU/device : {cfg.DEVICE}")
    print(f"  Batch size : {cfg.N_COLLOCATION}")
    print(f"  FEM/call   : {t_single:.2f}s")
    print(f"  GT (64 sim): {t_gt:.0f}s")
    print(f"  FMD 4-call : {t_fmd:.0f}s")
    print("="*55)

    if t_single > 5:
        print("\n⚠  FEM calls đang chậm (>5s). Thử giảm FEM_NX=20 trong config.py")
    if cfg.DEVICE == "cpu":
        print(f"\n⚠  Đang chạy trên CPU — GPU chưa được nhận diện.")
        print("   Kiểm tra: pip install torch --index-url https://download.pytorch.org/whl/cu118")

    print("\nCó thể chạy pipeline đầy đủ:")
    print("  python run_01_ground_truth.py")


if __name__ == "__main__":
    main()
