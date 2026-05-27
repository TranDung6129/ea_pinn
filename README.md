# FMD-PINN Benchmark 1 — 2D Reaction-Diffusion
**PDE:** ∂u/∂t = D∇²u + αu − βu³ | **Failure:** E(u) = max u − 1.5

---

## Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 6-core (12-thread) | Same |
| GPU | NVIDIA T2000 4GB | GTX 1070 8GB |
| RAM | 16 GB | 32 GB |
| Disk | 2 GB | 5 GB |

---

## Setup (one-time, ~5 min)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Quick Test (10 min, before full run)
Tests the full pipeline with tiny settings:
```bash
python run_04_fmd_pinn.py --quick
```
Expected output: A5_full_fmd runs 30 oracle calls, saves phase diagram.

---

## Full Pipeline (run in order)

### STEP 1 — Ground Truth (~47 min on 6 cores)
```bash
python run_01_ground_truth.py
```
**What runs:** 15³ = 3,375 FEM simulations in parallel (6 workers)
**Output:** `results/ground_truth.npz`, `results/phase_diagram_slices.png`
**Checkpoint CP1:** Console prints "collapse fraction: 0.3–0.6"

---

### STEP 2 — Vanilla PINN Baseline (~30 min)
```bash
python run_02_vanilla_pinn.py
```
**What runs:** Static PINN on 3 configs × 5 seeds
**Output:** `results/vanilla_pinn_*.json`, `results/vanilla_pinn_summary.json`
**Checkpoint CP3:** Near-boundary False Safe Rate should be HIGH (>20%)
This documents that vanilla PINN fails — expected behaviour.

---

### STEP 3 — Bayesian Optimization Baseline (~1 h)
```bash
python run_03_bayesian_opt.py
```
**What runs:** GP+EI acquisition × 120 FEM calls × 5 seeds
**Output:** `results/bo_fem_*.json`, `results/bo_fem_summary.json`
**Note:** This is the SOTA baseline to beat.

---

### STEP 4 — FMD-PINN + Ablation Studies (~4–8 h)
```bash
# Full run (all variants, 5 seeds each)
python run_04_fmd_pinn.py

# Single variant (faster)
python run_04_fmd_pinn.py --variant A5_full_fmd --seeds 2

# Available variants:
#   A2_event_only    — event loss only (no min-max)
#   A3_adaptive_only — adaptive sampling only
#   A4_no_minmax     — event + adaptive, passive p-space
#   A5_full_fmd      — full proposed method (default)
```
**Output:** `results/A5_full_fmd_seed*_metrics.json`, `checkpoints/*.pt`
**Checkpoint CP4:** FMD-PINN N_δ < BO N_δ (main claim)

---

### STEP 5 — Final Evaluation (~5 min)
```bash
python run_05_evaluate.py
```
**What runs:** Loads all results, computes all metrics, generates paper figures
**Output:**
- `results/oracle_efficiency.png` — main efficiency claim figure
- `results/fsr_curves.png`        — safety metric figure
- `results/field_*.png`           — solution accuracy figures
- `results/final_report.json`     — full metric table

---

## Output Files Summary

```
results/
├── ground_truth.npz              ← dense FEM phase diagram
├── phase_diagram_slices.png      ← Figure: ground truth ∂C
├── vanilla_pinn_summary.json     ← B2 baseline results
├── bo_fem_summary.json           ← B3 BO+FEM results
├── ablation_summary.json         ← A1–A5 comparison
├── oracle_efficiency.png         ← Figure: main claim
├── fsr_curves.png                ← Figure: safety metric
├── field_*.png                   ← Figure: solution accuracy
└── final_report.json             ← full metric table

checkpoints/
└── fmd_pinn_seed*_minmax1_ev1_as1.pt   ← saved PINN weights
```

---

## Expected Results

| Method | N_δ (oracle calls) | FSR | δ_H |
|--------|-------------------|-----|-----|
| FEM Grid Search | ~3375 (15³) | — | 0 |
| BO + FEM | ~80–120 | ~3% | ~0.08 |
| FMD-PINN (full) | ~30–60 | <2% | ~0.06 |
| FMD no min-max | ~80–100 | <3% | ~0.09 |
| Vanilla PINN | N/A | >20% | N/A |

---

## Troubleshooting

**CUDA out of memory:**
```python
# In config.py, reduce:
N_COLLOCATION = 1024    # default 2048
SPATIAL_HIDDEN = [64, 64, 64, 64]  # default [128,128,128,128]
```

**FEM too slow:**
```python
# In config.py, reduce:
FEM_NX = 24    # default 32 (coarser but 3× faster)
```

**Min-max oscillates:**
```python
# In config.py, reduce:
OUTER_LR = 0.02       # default 0.04
GRAD_CLIP = 0.5       # default 1.0
```

**Quick smoke test (2 min):**
```bash
python -c "
from src.fem_oracle import solve_reaction_diffusion
r = solve_reaction_diffusion(8.0, 1.0, 0.01)
print('FEM OK  E=', r['E'])

import torch
from src.pinn_model import build_model
m = build_model()
x = torch.rand(64,3)
p = torch.rand(3)
print('PINN OK  out=', m(x,p).shape)
"
```
