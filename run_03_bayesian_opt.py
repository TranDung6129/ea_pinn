"""
run_03_bayesian_opt.py — hai baseline BO, cùng thước đo với FMD-PINN.

  bo_max      : BO gốc, tối ưu max E. Đây là BO "đúng sách", và nó cho thấy
                vì sao không lấy thẳng một công cụ tối ưu hoá làm công cụ dựng
                biên: nó dồn ngân sách vào sâu miền sập, GP dự đoán dương khắp
                nơi, tập mức không rỗng, δ_H ra vô cùng.
  bo_boundary : cùng GP + EI + oracle + ngân sách, nhưng mục tiêu là min |E|.
                Đây mới là baseline công bằng cho bài toán dựng biên.

Cả hai đo bằng đúng cách đo của FMD-PINN: biên suy từ tập mức không của
surrogate trên cùng lưới ground truth, khoảng cách bằng boundary_distances.

  python run_03_bayesian_opt.py                 # chạy cả hai
  python run_03_bayesian_opt.py --which boundary
"""
import sys, os, time, json, argparse, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import freeze_support
freeze_support()

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

import numpy as np
import config as cfg
from src.baselines import BayesianOptBaseline
from src.bo_boundary import BoundaryBO
from src.ground_truth import generate_ground_truth
from src.metrics import compute_all_metrics, bo_style_surrogate


def run_baseline(kind: str, gt: dict) -> dict:
    print("\n" + "=" * 60)
    print(f"BASELINE {kind}   budget={cfg.ORACLE_BUDGET}  seeds={cfg.N_SEEDS}")
    print("=" * 60)

    prefix = "bo_fem" if kind == "bo_max" else "bo_boundary"
    all_m = []
    for i in range(cfg.N_SEEDS):
        print(f"\n── seed {i+1}/{cfg.N_SEEDS}")
        t0 = time.time()

        # Lịch sử oracle đã có thì KHÔNG chạy lại BO. Lịch sử không phụ thuộc
        # ground truth, chỉ metrics mới phụ thuộc. Nhờ vậy đổi lưới GT rồi tính
        # lại metrics không mất thêm 20 phút chạy BO.
        hist_path = os.path.join(cfg.RESULTS_DIR,
                                 f"{prefix}_seed{cfg.BASE_SEED + i}.json")
        if os.path.exists(hist_path) and not FORCE:
            res = json.load(open(hist_path))
            print(f"  nạp lại lịch sử có sẵn: {os.path.basename(hist_path)} "
                  f"({len(res['oracle_history'])} lời gọi)")
        else:
            cls = BayesianOptBaseline if kind == "bo_max" else BoundaryBO
            res = cls(oracle_budget=cfg.ORACLE_BUDGET,
                      seed=cfg.BASE_SEED + i, n_initial_points=10).run()
        dt = time.time() - t0

        # bo_style_surrogate(history, params): khớp GP trên lịch sử rồi dự đoán
        # tại params. compute_all_metrics gọi nó với lưới đo 24^3 để trích biên
        # và với lưới GT để tính FSR/CR — hai lưới khác nhau, có chủ ý.
        m = compute_all_metrics(
            oracle_history=res["oracle_history"], gt=gt,
            t_train_gpu_h=0.0, delta_target=None,
            n_grid=cfg.METRIC_GRID,
            E_pred_fn=bo_style_surrogate)
        m.update({"seed": i, "method": kind, "wall_clock_actual_s": dt})
        all_m.append(m)

        n_col = sum(1 for r in res["oracle_history"] if r["E_true"] > 0)
        n_edge = sum(1 for r in res["oracle_history"] if abs(r["E_true"]) < 0.3)
        print(f"  δ_H[{m['dh_metric']}]={m['hausdorff_final']:.4f}  "
              f"chamfer={m['dh_chamfer']:.4f}  N_δ={m['n_delta']}")
        print(f"  FSR={m['fsr_band']*100:.1f}%  báo động giả={m['far_band']*100:.1f}%"
              f"  lệch={m['bias_band']:+.3f}  CR={m['coverage_ratio']*100:.1f}%")
        print(f"  lời gọi: {n_col}/{len(res['oracle_history'])} sập, "
              f"{n_edge} nằm trong dải biên |E|<0.3   ({dt/60:.1f} phút)")

    def agg(k):
        v = [m[k] for m in all_m if m.get(k) is not None
             and np.isfinite(m[k]) and m[k] != -1]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)

    print(f"\n── {kind} trung bình ──")
    for k in ["hausdorff_final", "dh_max", "dh_chamfer", "n_delta",
              "fsr_band", "far_band", "coverage_ratio"]:
        mu, sd = agg(k)
        print(f"  {k:<18} {mu:.4f} ± {sd:.4f}")

    summary = {"method": kind, "oracle_budget": cfg.ORACLE_BUDGET,
               "dh_metric": all_m[0]["dh_metric"], "metrics_per_seed": all_m,
               "hausdorff_mean": agg("hausdorff_final")[0],
               "hausdorff_std": agg("hausdorff_final")[1],
               "n_delta_mean": agg("n_delta")[0]}
    fname = "bo_fem_summary.json" if kind == "bo_max" else "bo_boundary_summary.json"
    out = os.path.join(cfg.RESULTS_DIR, fname)
    json.dump(summary, open(out, "w"), indent=2, default=float)
    print(f"\n✓ {out}")
    return summary


FORCE = False


def main():
    global FORCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["max", "boundary", "both"],
                    default="both")
    ap.add_argument("--force", action="store_true",
                    help="chạy lại BO từ đầu thay vì nạp lịch sử đã có")
    args = ap.parse_args()
    FORCE = args.force

    gt = generate_ground_truth()
    from src.metrics import resolution_floor
    print(f"  ground truth {gt.get('n_grid')}^3 · lưới đo {cfg.METRIC_GRID}^3 · "
          f"sàn {resolution_floor(int(gt.get('n_grid', cfg.GT_GRID)), cfg.METRIC_GRID):.3f} · "
          f"δ_target {getattr(cfg, 'DELTA_TARGET', None)}")

    if args.which in ("max", "both"):
        run_baseline("bo_max", gt)
    if args.which in ("boundary", "both"):
        run_baseline("bo_boundary", gt)

    print("\nTiếp theo: python run_05_evaluate.py --tag cap")


if __name__ == "__main__":
    main()