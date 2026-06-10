"""
diag_fsr.py — diagnose where/why false-safes accumulate.
Reads one seed's full oracle history JSON and reports:
  • false-safe events over time (early vs late)
  • where in parameter space they cluster
  • E_pinn vs E_true gap as training progresses
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config as cfg

SEED = 46   # worst seed (seed index 4, FSR 23.5%); change as needed
name = f"fmd_pinn_seed{SEED}_minmax1_ev1_as1.json"
path = os.path.join(cfg.RESULTS_DIR, name)

with open(path) as f:
    data = json.load(f)
hist = data["oracle_history"]
print(f"Loaded {len(hist)} oracle calls from seed {SEED}\n")

# False-safe events: E_pinn < 0 but E_true > 0
fs = [(r["oracle_call"], r["alpha"], r["beta"], r["D"],
       r["E_pinn"], r["E_true"], r.get("is_exploit"))
      for r in hist
      if r["E_pinn"] < 0 and r["E_true"] > 0]

print(f"Total false-safe events: {len(fs)}")
early = [e for e in fs if e[0] < 100]
late  = [e for e in fs if e[0] >= 100]
print(f"  early (call <100): {len(early)}")
print(f"  late  (call>=100): {len(late)}   <- if >> early, suggests forgetting\n")

print("Late false-safe events (call, alpha, beta, D, E_pinn, E_true, exploit):")
for e in late:
    print(f"  call={e[0]:3d}  a={e[1]:5.2f} b={e[2]:4.2f} D={e[3]:.4f}  "
          f"E_pinn={e[4]:+.3f} E_true={e[5]:+.3f}  exploit={e[6]}")

# How exploit vs explore relate to false-safes
exploit_fs = [e for e in fs if e[6]]
explore_fs = [e for e in fs if not e[6]]
print(f"\nFalse-safes by phase: exploit={len(exploit_fs)}  explore={len(explore_fs)}")

# E gap (|E_pinn - E_true|) trend in windows of 25 calls
print("\nMean |E_pinn - E_true| per 25-call window:")
for lo in range(0, len(hist), 25):
    w = hist[lo:lo+25]
    gap = np.mean([abs(r["E_pinn"] - r["E_true"]) for r in w])
    coll = np.mean([1 for r in w if r["E_true"] > 0]) if w else 0
    print(f"  calls {lo:3d}-{lo+24:3d}:  gap={gap:.3f}  "
          f"n_collapse_visited={sum(1 for r in w if r['E_true']>0)}/{len(w)}")