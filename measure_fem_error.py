import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config as cfg
from src.fem_oracle import solve_reaction_diffusion

def measure_fem_error():
    print("="*60)
    print("FEM ORACLE ERROR MEASUREMENT")
    print("Comparing E_true between NX=15 (current) and NX=30")
    print("="*60)
    
    # Representative boundary/collapse points
    pts = [
        (4.0, 4.0, 0.007, "near_boundary_high_beta"),
        (5.0, 2.5, 0.01,  "near_boundary_mid_beta"),
        (6.0, 1.5, 0.01,  "near_boundary_low_beta"),
        (3.0, 3.0, 0.15,  "near_boundary_high_D"),
        (10.0, 1.0, 0.01, "collapse"),
        (14.0, 0.5, 0.005,"extreme_collapse"),
        (2.0, 4.0, 0.1,   "stable"),
    ]
    
    diffs = []
    
    for a, b, D, label in pts:
        # Run with default NX=15
        cfg.FEM_NX = 15
        r15 = solve_reaction_diffusion(a, b, D)
        e15 = r15["E"]
        
        # Run with high-res NX=30
        cfg.FEM_NX = 30
        r30 = solve_reaction_diffusion(a, b, D)
        e30 = r30["E"]
        
        diff = abs(e15 - e30)
        diffs.append(diff)
        
        print(f"[{label:25s}] α={a:4.1f} β={b:4.1f} D={D:6.4f} | E_15={e15:+.4f}  E_30={e30:+.4f} | Diff = {diff:.4f}")
        
    print("-" * 60)
    print(f"Mean Error: {np.mean(diffs):.4f}")
    print(f"Max Error:  {np.max(diffs):.4f}")
    print(f"Recommended τ ≈ {np.max(diffs) * 1.5:.4f}")
    
if __name__ == "__main__":
    measure_fem_error()
