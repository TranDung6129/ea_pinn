"""
recompute_fsr.py
----------------
Recompute FSR (and fsr_curve) with the new dead-band tau, in place,
WITHOUT retraining anything. Reads the full oracle history that already
exists in results/fmd_pinn_seed*.json and patches the corresponding
*_seedN_metrics.json files.

Mapping of variant → (minmax, ev, as) flag suffix:
  A2_event_only    : minmax0_ev1_as0
  A3_adaptive_only : minmax0_ev0_as1
  A4_no_minmax     : minmax0_ev1_as1
  A5_full_fmd      : minmax1_ev1_as1

seed index i → physical seed (cfg.BASE_SEED + i)
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config as cfg

TAU = 0.05

VARIANTS = {
    "A2_event_only":    "minmax0_ev1_as0",
    "A3_adaptive_only": "minmax0_ev0_as1",
    "A4_no_minmax":     "minmax0_ev1_as1",
    "A5_full_fmd":      "minmax1_ev1_as1",
}


def fsr(hist, tau=TAU):
    true_collapse = [r for r in hist if r.get("E_true", 0) > tau]
    if not true_collapse:
        return 0.0
    fs = [r for r in hist
          if r.get("E_pinn", -1) < 0 and r.get("E_true", 0) > tau]
    return len(fs) / len(true_collapse)


def fsr_curve(hist, tau=TAU):
    return [fsr(hist[:i], tau) for i in range(1, len(hist) + 1)]


def main():
    print(f"[recompute_fsr] tau = {TAU}\n")
    for variant, suffix in VARIANTS.items():
        print(f"── {variant} ──")
        for i in range(cfg.N_SEEDS):
            phys = cfg.BASE_SEED + i
            hist_file = os.path.join(
                cfg.RESULTS_DIR, f"fmd_pinn_seed{phys}_{suffix}.json")
            metr_file = os.path.join(
                cfg.RESULTS_DIR, f"{variant}_seed{i}_metrics.json")

            if not os.path.exists(hist_file):
                print(f"  seed{i}: history file missing ({os.path.basename(hist_file)}) — skip")
                continue
            if not os.path.exists(metr_file):
                print(f"  seed{i}: metrics file missing — skip")
                continue

            with open(hist_file) as f:
                hist = json.load(f)["oracle_history"]
            with open(metr_file) as f:
                m = json.load(f)

            old_fsr = m.get("fsr")
            new_fsr = fsr(hist)
            m["fsr"]       = new_fsr
            m["fsr_curve"] = fsr_curve(hist)
            m["fsr_tau"]   = TAU

            with open(metr_file, "w") as f:
                json.dump(m, f, indent=2, default=float)

            print(f"  seed{i}: FSR {old_fsr*100:5.1f}% → {new_fsr*100:5.1f}%")
        print()


if __name__ == "__main__":
    main()