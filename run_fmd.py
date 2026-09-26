"""
run_fmd.py — chạy FMD-PINN và ablation. Thay cho run_04_fmd_pinn.py.

Vì sao thay: run_04 có lỗi --force-restart không ngăn được trainer chạy tiếp
từ history JSON cũ, và nó truyền delta_target=0.1 ở hai chỗ, mốc nằm dưới sàn
rời rạc hoá nên N_delta luôn bằng -1.

Ablation mới bám đúng phát biểu bài toán E(p)=0:
  B1_passive  : lấy mẫu LHS thụ động                       (không chủ động)
  B2_levelset : đề xuất trên tập mức không của PINN
  B3_bisect   : levelset + bisection trên oracle
  B4_full     : + ưu tiên cấu hình tới hạn (||grad_p E|| suy biến)

Cách chạy:
  python run_fmd.py --variant B4_full --seeds 1          # pilot một seed
  python run_fmd.py                                       # tất cả, 5 seed
  python run_fmd.py --variant B4_full --force-restart
"""

import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import freeze_support
freeze_support()

import numpy as np
import config as cfg
from src.fmd_pinn import FMDPINNTrainer
from src.ground_truth import generate_ground_truth
from src.metrics import compute_all_metrics

VARIANTS = {
    "B1_passive":  dict(use_levelset=False, use_bisection=False, use_knee=False),
    "B2_levelset": dict(use_levelset=True,  use_bisection=False, use_knee=False),
    "B3_bisect":   dict(use_levelset=True,  use_bisection=True,  use_knee=False),
    "B4_full":     dict(use_levelset=True,  use_bisection=True,  use_knee=True),
}


def run_one(name, flags, seed_idx, gt, force_restart=False, tag=""):
    phys = cfg.BASE_SEED + seed_idx
    label = f"{name}_{tag}" if tag else name
    out = os.path.join(cfg.RESULTS_DIR, f"{label}_seed{seed_idx}_metrics.json")
    if os.path.exists(out) and not force_restart:
        print(f"  [{label}] seed={seed_idx} → đã có metrics, nạp lại.")
        return json.load(open(out))

    trainer = FMDPINNTrainer(seed=phys, tag=tag, **flags)
    if force_restart:
        FMDPINNTrainer.clear_state(trainer.name)
    else:
        trainer.load_checkpoint()

    t0 = time.time()
    results = trainer.run()
    dt = time.time() - t0

    m = compute_all_metrics(
        oracle_history=results["oracle_history"],
        gt=gt,
        t_train_gpu_h=dt / 3600,
        delta_target=None,                 # tự tính theo sàn rời rạc hoá
        model=trainer.model,
        ckpt_name_base=trainer.name,
        n_grid=cfg.METRIC_GRID,
    )
    m.update({"variant": label, "seed": seed_idx, "dt_total_s": dt,
              "sup_mode": cfg.SUP_MODE, "hidden": list(cfg.SPATIAL_HIDDEN),
              "n_fourier": cfg.N_FOURIER, "eval_nt": cfg.EVAL_NT})
    with open(out, "w") as f:
        json.dump(m, f, indent=2, default=float)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all",
                    choices=list(VARIANTS) + ["all"])
    ap.add_argument("--seeds", type=int, default=cfg.N_SEEDS)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--force-restart", action="store_true")
    ap.add_argument("--tag", default="",
                    help="hậu tố cho tên file, để hai thí nghiệm không đè nhau")
    ap.add_argument("--sup", choices=["regress", "hinge"], default=None,
                    help="chế độ loss giám sát, ghi đè cfg.SUP_MODE")
    ap.add_argument("--hidden", type=int, default=None,
                    help="bề rộng mỗi lớp ẩn, ví dụ 128")
    ap.add_argument("--layers", type=int, default=None, help="số lớp ẩn")
    ap.add_argument("--fourier", type=int, default=None,
                    help="số Fourier feature, ví dụ 64")
    ap.add_argument("--eval-nt", type=int, default=None,
                    help="số lát thời gian trong lưới đánh giá E")
    ap.add_argument("--region", default=None,
                    help='vùng được phép gọi oracle, ví dụ "beta:0.4,1.0" '
                         'hoặc "D:0,0.65" (toạ độ chuẩn hoá [0,1])')
    args = ap.parse_args()

    if args.budget:
        cfg.ORACLE_BUDGET = args.budget
    if args.eval_nt:
        cfg.EVAL_NT = args.eval_nt
        import src.sampling as _s
        _s._EVAL_NT = args.eval_nt
        _s._EVAL_CACHE.clear()
    if args.sup:
        cfg.SUP_MODE = args.sup
    if args.hidden or args.layers:
        w = args.hidden or cfg.SPATIAL_HIDDEN[0]
        n = args.layers or len(cfg.SPATIAL_HIDDEN)
        cfg.SPATIAL_HIDDEN = [w] * n
    if args.fourier:
        cfg.N_FOURIER = args.fourier
    if args.region:
        axis, rng = args.region.split(":")
        a, b = (float(x) for x in rng.split(","))
        cfg.ORACLE_REGION = {axis.strip(): (a, b)}

    print("\n" + "=" * 60)
    print(f"FMD-PINN  device={cfg.DEVICE}  budget={cfg.ORACLE_BUDGET}")
    print(f"  sup={cfg.SUP_MODE}  hidden={cfg.SPATIAL_HIDDEN}  "
          f"fourier={cfg.N_FOURIER}  eval_nt={cfg.EVAL_NT}"
          + (f"  tag={args.tag}" if args.tag else ""))
    if getattr(cfg, "ORACLE_REGION", None):
        print(f"  vùng gọi oracle: {cfg.ORACLE_REGION}  "
              f"(ràng buộc PDE vẫn phủ toàn không gian)")
    print("=" * 60)

    gt = generate_ground_truth()
    chosen = VARIANTS if args.variant == "all" else {args.variant: VARIANTS[args.variant]}

    table = {}
    for name, flags in chosen.items():
        label = f"{name}_{args.tag}" if args.tag else name
        print(f"\n{'='*50}\n{label}\n{'='*50}")
        ms = [run_one(name, flags, s, gt, args.force_restart, args.tag)
              for s in range(args.seeds)]
        table[label] = ms
        for m in ms:
            print(f"  seed={m['seed']}  δ_H[{m['dh_metric']}]={m['hausdorff_final']:.4f}  "
                  f"(đuôi {m['hausdorff_tail_mean']:.4f}±{m['hausdorff_tail_std']:.4f})  "
                  f"N_δ={m['n_delta']}  CR={m['coverage_ratio']*100:.1f}%")
            print(f"      biên: max={m['dh_max']:.4f}  p95={m['dh_p95']:.4f}  "
                  f"chamfer={m['dh_chamfer']:.4f}  "
                  f"({m['n_pred_boundary']} điểm dự đoán / {m['n_true_boundary']} thật)")
            print(f"      theo vùng: knee(β<{m['knee_beta']}) δ_H={m['dh_knee']:.4f} "
                  f"chamfer={m['chamfer_knee']:.4f} FSR={m['fsr_knee']*100:.1f}% "
                  f"BĐgiả={m['far_knee']*100:.1f}% lệch={m['bias_knee']:+.3f}  |  "
                  f"còn lại δ_H={m['dh_rest']:.4f} chamfer={m['chamfer_rest']:.4f}")
            if m.get("dh_unseen") is not None:
                print(f"      NGOẠI SUY: trong vùng δ_H={m['dh_seen']:.4f} "
                      f"chamfer={m['chamfer_seen']:.4f} (n={m['n_seen']})  |  "
                      f"NGOÀI vùng δ_H={m['dh_unseen']:.4f} "
                      f"chamfer={m['chamfer_unseen']:.4f} (n={m['n_unseen']})")
            print(f"      dải biên |E|<{m['fsr_band_limit']}:  "
                  f"FSR={m['fsr_band']*100:.1f}% (n={m['n_band']})   "
                  f"báo động giả={m['far_band']*100:.1f}% (n={m['n_band_safe']})   "
                  f"lệch E_pinn-E_true={m['bias_band']:+.3f}")

    print("\n" + "=" * 86)
    print(f"{'Variant':<14}{'δ_H':>9}{'±ckpt':>9}{'N_δ':>7}"
          f"{'FSR_biên':>10}{'FSR_lưới':>10}{'CR':>8}{'giờ':>8}")
    print("-" * 86)
    summary = {}
    for name, ms in table.items():
        def mean(k):
            v = [m[k] for m in ms if m.get(k) is not None
                 and np.isfinite(m[k]) and m[k] != -1]
            return float(np.mean(v)) if v else float("nan")
        row = dict(hausdorff=mean("hausdorff_final"),
                   tail_std=mean("hausdorff_tail_std"),
                   n_delta=mean("n_delta"),
                   fsr_band=mean("fsr_band"),
                   fsr_grid=mean("fsr_grid"),
                   cr=mean("coverage_ratio"),
                   hours=mean("dt_total_s") / 3600 if ms else float("nan"))
        summary[name] = row
        print(f"{name:<14}{row['hausdorff']:>9.4f}{row['tail_std']:>9.4f}"
              f"{row['n_delta']:>7.0f}{row['fsr_band']*100:>9.1f}%"
              f"{row['fsr_grid']*100:>9.1f}%{row['cr']*100:>7.1f}%"
              f"{row['hours']:>8.2f}")
    print("=" * 86)
    d0 = table[list(table)[0]][0]
    print(f"δ_target = {d0['delta_target']:.3f}   "
          f"sàn rời rạc hoá = {d0['delta_floor']:.3f}   τ = {d0['fsr_tau']}")
    print(f"FSR_biên: chỉ xét dải sát biên τ < E_true < {d0['fsr_band_limit']}, "
          f"đo trên lưới GT — đây là chỉ số so sánh được.")

    with open(os.path.join(cfg.RESULTS_DIR, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n✓ Đã lưu ablation_summary.json")


if __name__ == "__main__":
    main()