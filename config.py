"""
config.py — Central configuration for FMD-PINN Benchmark 1
PDE: ∂u/∂t = D∇²u + αu − βu³  on  Ω=[0,1]², t∈[0,1]
BC : Neumann (zero-flux)
IC : u(x,y,0) = 0.1·sin(πx)·sin(πy)
E(u) = max_{x,t} u(x,y,t) − u_threshold   (blow-up indicator)
"""

import torch
import os

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CPU_WORKERS = 6          # match your 6-core CPU

# Auto-detect VRAM and set safe batch size
def _get_safe_batch():
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 5.0:        # T2000 4 GB
            return 2048
        else:                     # GTX 1070 8 GB
            return 4096
    return 1024                   # CPU fallback

N_COLLOCATION = _get_safe_batch()

# ── Problem Definition ────────────────────────────────────────────────────────
# Parameter space  p = (α, β, D)
ALPHA_RANGE = (1.0, 15.0)    # reaction rate
BETA_RANGE  = (0.5,  5.0)    # nonlinear damping
D_RANGE     = (0.005, 0.3)   # diffusion coefficient  (log-scale internally)

U_THRESHOLD = 1.5            # failure blow-up threshold

# Domain
T_END  = 1.0
DOMAIN = (0.0, 1.0)          # x,y both in [0,1]

# ── FEM Oracle (scipy method-of-lines) ───────────────────────────────────────
FEM_NX   = 32       # spatial grid per dimension (32×32 = 1024 DOFs → ~0.5s/sim)
FEM_NT   = 100      # saved time snapshots (reduced from 200 for speed)
FEM_RTOL = 1e-5
FEM_ATOL = 1e-6

# ── PINN Architecture ─────────────────────────────────────────────────────────
N_FOURIER      = 32          # Fourier features for (x,y,t) → 64-dim after sin+cos
FOURIER_SCALE  = 2.0         # σ of random Fourier matrix
SPATIAL_HIDDEN = [64, 64, 64, 64]   # thu nhỏ spatial branch
PARAM_HIDDEN   = [128, 128, 128]    # tăng param branch
FILM_DIM       = 128                    # FiLM conditioning dimension

# ── Training ─────────────────────────────────────────────────────────────────
N_BOUNDARY   = 256
N_INITIAL    = 256
N_EVENT      = 512           # extra points near failure surface

LR_ADAM      = 3e-4
ADAM_EPOCHS  = 8000
LBFGS_STEPS  = 10
LBFGS_HISTORY= 20

LAMBDA_BC    = 10.0
LAMBDA_IC    = 10.0
LAMBDA_EVENT = 1.0           # initial; increases toward LAMBDA_EVENT_MAX
LAMBDA_EVENT_MAX = 100.0

# ── Min-Max Adversarial Loop ──────────────────────────────────────────────────
ORACLE_BUDGET = 200
EXPLOIT_RATIO      = 0.80    # 80% exploitation, 20% exploration
OUTER_LR           = 0.04    # gradient-ascent step in normalised [0,1]³
OUTER_STEPS        = 15      # ascent steps per oracle call
GRAD_CLIP          = 1.0
FSR_ALERT_THRESH   = 0.05    # trigger re-allocation if FSR > 5 %

# ── Ground Truth ──────────────────────────────────────────────────────────────
GT_GRID      = 15            # 15³ = 3375 FEM runs for dense phase diagram
GT_SEEDS     = 1             # deterministic IC for GT

# ── Ablation ─────────────────────────────────────────────────────────────────
N_SEEDS   = 5                # repeat each experiment with N_SEEDS random seeds
BASE_SEED = 42               # base random seed (seed_i = BASE_SEED + i)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
CKPT_DIR      = os.path.join(BASE_DIR, "checkpoints")
GT_CACHE      = os.path.join(RESULTS_DIR, "ground_truth.npz")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

print(f"[config] device={DEVICE}  batch={N_COLLOCATION}  oracle_budget={ORACLE_BUDGET}")
