"""
run_05_evaluate.py — bảng so sánh cuối + hình. Chỉ đọc results/.

  python run_05_evaluate.py
  python run_05_evaluate.py --tag cap
"""
import sys, os, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import config as cfg

from src.ground_truth import generate_ground_truth
from src.metrics import normalise_for_hausdorff, resolution_floor
import plot_metric

VARIANTS = ["B1_passive", "B2_levelset", "B3_bisect", "B4_full"]
LABEL = {"B1_passive": "B1 thụ động", "B2_levelset": "B2 tập mức không",
         "B3_bisect": "B3 + bisection", "B4_full": "B4 + knee"}


def seed_metrics(variant, tag):
    pat = f"{variant}_{tag}_seed*_metrics.json" if tag else f"{variant}_seed*_metrics.json"
    return [json.load(open(f))
            for f in sorted(glob.glob(os.path.join(cfg.RESULTS_DIR, pat)))]


def agg(ms, key):
    v = [m[key] for m in ms if m.get(key) is not None
         and np.isfinite(m[key]) and m[key] != -1]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ CUỐI" + (f"   tag={args.tag}" if args.tag else ""))
    print("=" * 60)

    gt = generate_ground_truth()
    mask = np.abs(gt["E_fem_flat"]) < 0.15
    floor = resolution_floor(int(gt.get("n_grid", cfg.GT_GRID)), cfg.METRIC_GRID)
    print(f"  biên thật: {int(mask.sum())} điểm   "
          f"sàn rời rạc hoá ≈ {floor:.3f}   chỉ số: {cfg.DH_METRIC}")

    rows = []
    for v in VARIANTS:
        ms = seed_metrics(v, args.tag)
        if not ms:
            continue
        rows.append((LABEL[v],) + tuple(
            agg(ms, k)[0] for k in ["hausdorff_final", "n_delta", "fsr_band",
                                    "far_band", "coverage_ratio"])
            + (agg(ms, "hausdorff_final")[1], agg(ms, "hausdorff_tail_std")[0],
               len(ms)))

    for fname, lab in [("bo_fem_summary.json", "BO (max E)"),
                       ("bo_boundary_summary.json", "BO (tìm biên)")]:
        p = os.path.join(cfg.RESULTS_DIR, fname)
        if not os.path.exists(p):
            continue
        ms = json.load(open(p)).get("metrics_per_seed", [])
        if ms:
            rows.append((lab,) + tuple(
                agg(ms, k)[0] for k in ["hausdorff_final", "n_delta", "fsr_band",
                                        "far_band", "coverage_ratio"])
                + (agg(ms, "hausdorff_final")[1], float("nan"), len(ms)))

    rows.append((f"FEM grid {cfg.GT_GRID}³", 0.0, float(cfg.GT_GRID ** 3),
                 0.0, 0.0, 1.0, 0.0, float("nan"), 1))

    print("\n" + "=" * 96)
    print(f"{'Phương pháp':<20}{'δ_H':>9}{'±seed':>8}{'±ckpt':>8}"
          f"{'N_δ':>7}{'FSR_b':>8}{'BĐgiả':>8}{'CR':>8}{'seeds':>7}")
    print("-" * 96)
    for n, dh, nd, fb, far, cr, sd, ts, k in rows:
        print(f"{n:<20}{dh:>9.4f}{sd:>8.4f}{ts:>8.4f}"
              f"{nd:>7.0f}{fb*100:>7.1f}%{far*100:>7.1f}%{cr*100:>7.1f}%{k:>7}")
    print("=" * 96)
    print("  N_δ: số lời gọi oracle để δ_H xuống dưới δ_target (không phải tổng).")
    print("  FEM grid là ground truth, không phải phương pháp: chi phí để có")
    print("  biên tham chiếu, đặt ở đây làm mốc so sánh chi phí.")
    print("  ±ckpt: dao động giữa 5 checkpoint cuối của cùng một seed.")

    print("\n" + "=" * 96)
    print("TÁCH THEO VÙNG   knee = β < %.1f" % cfg.KNEE_BETA)
    print(f"{'Phương pháp':<20}{'δ_H knee':>10}{'cham knee':>11}"
          f"{'FSR knee':>10}{'BĐgiả knee':>12}{'lệch knee':>11}"
          f"{'δ_H còn lại':>13}{'cham còn lại':>14}")
    print("-" * 96)
    for v in VARIANTS:
        ms = seed_metrics(v, args.tag)
        if not ms or not np.isfinite(ms[0].get("dh_knee", np.nan)):
            continue
        g = lambda k: agg(ms, k)[0]
        print(f"{LABEL[v]:<20}{g('dh_knee'):>10.4f}{g('chamfer_knee'):>11.4f}"
              f"{g('fsr_knee')*100:>9.1f}%{g('far_knee')*100:>11.1f}%"
              f"{g('bias_knee'):>11.3f}{g('dh_rest'):>13.4f}"
              f"{g('chamfer_rest'):>14.4f}")
    print("=" * 96)

    print("\n── Hình ──")
    plot_metric.TAG = args.tag
    plot_metric.plot_hausdorff()
    plot_metric.plot_safety()

    json.dump({r[0]: dict(hausdorff=r[1], n_delta=r[2], fsr_band=r[3],
                          far_band=r[4], cr=r[5], std_seed=r[6],
                          std_ckpt=r[7], n_seeds=r[8]) for r in rows},
              open(os.path.join(cfg.RESULTS_DIR, "final_report.json"), "w"),
              indent=2, default=float)
    print(f"\n✓ {cfg.RESULTS_DIR}/final_report.json")


if __name__ == "__main__":
    main()