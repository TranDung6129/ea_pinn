"""
config.py — Cấu hình trung tâm, FMD-PINN Benchmark 1
PDE: du/dt = D*lap(u) + alpha*u - beta*u^3  trên Omega=[0,1]^2, t in [0,1]
BC : Neumann (zero-flux)
IC : u(x,y,0) = 0.1*sin(pi x)*sin(pi y)
E(u;p) = max_{x,t} u - u_thr

Bài toán: dựng mặt nghiệm dC = { p : E(p) = 0 }.
"""

import os

if hasattr(os, "add_dll_directory"):
    _lib_bin = r"pinn_venv_new\library\bin"
    os.environ["PATH"] = os.pathsep.join(
        p for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and _lib_bin not in p.lower().replace("/", "\\"))
    _torch_lib = r'C:\Users\ADMIN\pinn_venv_new\Lib\site-packages\torch\lib'
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

import torch

# ── Phần cứng ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CPU_WORKERS = max(1, int((os.cpu_count() or 1) * 0.6))

# ── Không gian tham số p = (alpha, beta, D) ──────────────────────────────────
ALPHA_RANGE = (1.0, 15.0)
BETA_RANGE  = (0.5,  5.0)
D_RANGE     = (0.005, 0.3)      # nội bộ dùng log-scale

U_THRESHOLD = 1.5
T_END  = 1.0
DOMAIN = (0.0, 1.0)

# ── Oracle FEM (scipy method-of-lines) ───────────────────────────────────────
# Không đổi: đã đo 0.119-0.155 s/lần gọi. Oracle KHÔNG phải nút thắt ở bài 2D,
# nên không động vào để mọi kết quả cũ còn so được.
FEM_NX   = 32
FEM_NT   = 100
FEM_RTOL = 1e-5
FEM_ATOL = 1e-6
FEM_DISCRETISATION_ERR = 0.0057   # đã đo bằng measure_fem_error.py (NX=15 vs 30)

# ── Kiến trúc PINN ───────────────────────────────────────────────────────────
N_FOURIER      = 32
FOURIER_SCALE  = 2.0
SPATIAL_HIDDEN = [128, 128, 128, 128]   # cấu hình đã chốt cho bảng 5 seed (tag "cap")

# ── Lưới đánh giá E (CỐ ĐỊNH) ────────────────────────────────────────────────
# E phải là hàm xác định của p. Lấy max trên mẫu ngẫu nhiên đổi mỗi lần gọi
# vừa gây nhiễu vừa lệch hệ thống về phía an toàn, tức thổi FSR lên.
EVAL_NX = 12
# EVAL_NT = 8. Đã thử 24 và BỊ BÁC BỎ; giữ ghi chú lại làm kết quả âm:
#   Giả thuyết: vùng knee kém vì lưới 8 lát bỏ sót đỉnh quá độ (diag_knee.py
#   đo được các ca sập trong knee đạt đỉnh ở t*/T ~0.83, sớm hơn phần còn lại
#   0.94-1.00; ở lát D nhỏ, 25% số ca knee đạt đỉnh trước cuối so với 0%
#   ngoài knee).
#   Thí nghiệm: B2_levelset, seed 42, hidden 128, EVAL_NT=24, 200 lời gọi.
#   Kết quả: δ_H 0.0723 -> 0.1642 · FSR dải biên 2.2% -> 49.9% · CR 99.4% ->
#   86.4% · lệch +0.044 -> -0.149. Checkpoint cuối (0.1642) xấu hơn trung bình
#   5 checkpoint cuối (0.0923 +/- 0.0382), tức mô hình đi ngược lên ở cuối.
#   δ_H vùng knee 0.1291, không khá hơn cấu hình 8 lát.
#   Đọc: E là max cứng nên gradient chỉ chảy qua một điểm argmax; lưới dày hơn
#   làm argmax nhảy giữa nhiều ứng viên hơn, tín hiệu giám sát nhiễu hơn, nhân
#   LAMBDA_SUP=5 thì mất ổn định.
#   Kết luận: độ phân giải thời gian KHÔNG phải nguyên nhân của vùng knee.
EVAL_NT = 8
SOFTMAX_BETA = 40.0     # chỉ dùng khi E_SMOOTH=True
# E là max cứng, đúng bằng định nghĩa của oracle FEM (max_u - u_thr), nên
# E_pinn và E_true là cùng một đại lượng. Với logsumexp thì E_smooth > E_hard
# một lượng phụ thuộc hình dạng trường, mạng bù bằng một hằng số trung bình
# nên sinh lệch âm có hệ thống ở đúng các ca sập.
E_SMOOTH     = False

# ── Huấn luyện ───────────────────────────────────────────────────────────────
N_PARAM_BATCH  = 32     # số tham số p mỗi bước — thay cho việc train 500 bước trên 1 p
# Trần số hàng mỗi lượt forward, để lô tự co theo EVAL_NT. Mỗi điểm tham số
# kéo theo cả lưới đánh giá (EVAL_NX^2 * EVAL_NT hàng). Card 4 GB thì để
# SCAN_ROWS quanh 200k.
SCAN_ROWS      = 400_000   # cho vòng quét có giữ đồ thị gradient
EVAL_ROWS      = 600_000   # cho các lượt no_grad khi đo
N_COLL_PER_P   = 256
N_BOUNDARY     = 256
N_INITIAL      = 256
N_SUP_BATCH    = 32
LAMBDA_BC      = 10.0
LAMBDA_IC      = 10.0
LAMBDA_SUP     = 5.0
SUP_MODE       = "regress"   # "regress" | "hinge". Hinge đã thử và kém hơn rõ:
                             # δ_H 0.1798 so với 0.0967 ở cùng seed và cấu hình.
SUP_MARGIN     = 0.1         # biên độ cho chế độ hinge
LR_ADAM        = 3e-4
INIT_STEPS     = 3000
STEPS_PER_CALL = 60
GRAD_CLIP      = 1.0

# ── Vòng truy tìm biên ───────────────────────────────────────────────────────
ORACLE_BUDGET = 200
N_SEED_CALLS  = 10      # LHS mồi để có cả điểm ổn định lẫn điểm sập
SCAN_GRID     = 16      # lưới quét tập mức không
SCAN_BAND     = 0.15
KNEE_WEIGHT   = 0.5     # trọng số cho số hạng ưu tiên theo độ dốc của biên
# Hướng ưu tiên. Dọc theo biên giải tích alpha = 2*pi^2*D + beta*u_thr^2, chuẩn
# gradient chuẩn hoá là 9.60 ở beta=0.6 và 1.44 ở beta=4.0. Vùng knee là chỗ
# gradient LỚN nhất, không phải nhỏ nhất.
#   "steep" thưởng cho gradient lớn, tức đi vào vùng knee. Mặc định.
#   "flat"  hành vi của bảng 5 seed đã chạy: thưởng cho gradient nhỏ, tức tránh
#           vùng knee. Giữ lại làm đối chứng; nó cho δ_H knee 0.0778 (tệ nhất
#           trong bốn biến thể) và báo động giả knee 42.1%.
KNEE_PREFER   = "steep"
BISECT_EVERY  = 3
CKPT_EVERY    = 10

# ── Đánh giá ─────────────────────────────────────────────────────────────────
FSR_TAU      = 0.01     # ~1.5x sai số rời rạc hoá FEM đã đo. Bản cũ để 0.05.
METRIC_GRID  = 24       # lưới quét biên dự đoán khi đo delta_H
# Cách đo khoảng cách biên dùng cho đường cong và N_delta:
#   "max"     Hausdorff cổ điển — do đúng một điểm tệ nhất quyết định, và vì
#             cả hai tập điểm nằm trên lưới cố định nên giá trị bị lượng tử
#             hoá. Đó là lý do δ_H đứng im ở 0.1843 qua nhiều mô hình khác hẳn.
#   "p95"     phân vị 95, bỏ qua điểm lạc. Mặc định.
#   "chamfer" khoảng cách trung bình hai chiều.
DH_METRIC    = "p95"
# Vùng knee dùng để tách chỉ số: beta nhỏ hơn ngưỡng này. Đây là chỗ biên gần
# như dựng đứng theo beta và ||grad_p E|| suy biến, tức là chỗ khó nhất.
KNEE_BETA    = 1.5
# Ngưỡng hội tụ dùng để tính N_delta. CỐ ĐỊNH, không suy ra từ sàn rời rạc hoá.
# Lý do: nếu để tự suy thì làm lưới GT mịn hơn sẽ kéo ngưỡng tụt theo, và mô
# hình tốt nhất (0.0708) sẽ không còn đạt, N_delta lại về -1 ở mọi biến thể.
# Mục đích của lưới GT mịn hơn là hạ SÀN để ngưỡng 0.080 nằm cách xa nhiễu của
# phép đo, chứ không phải để hạ ngưỡng.
#   GT 15^3 + lưới đo 24^3 -> sàn 0.062, ngưỡng bằng 1.3 lần sàn (quá sát)
#   GT 25^3 + lưới đo 24^3 -> sàn 0.038, ngưỡng bằng 2.1 lần sàn
DELTA_TARGET = 0.080
# Dải ngưỡng phụ để báo cáo N_delta kèm, cho thấy thứ hạng giữa các biến thể
# có ổn định theo ngưỡng hay không. Một con số N_delta duy nhất là chỗ phản
# biện sẽ nhắm vào trước tiên.
DELTA_SWEEP  = [0.06, 0.07, 0.08, 0.10, 0.12]

# ── Phép thử ngoại suy ───────────────────────────────────────────────────────
# Giới hạn LỜI GỌI ORACLE vào một hộp con của không gian tham số, theo toạ độ
# chuẩn hoá [0,1] trên từng trục (alpha, beta tuyến tính; D theo thang log).
# Ràng buộc PDE vẫn phủ toàn bộ không gian: điểm collocation không bị giới hạn.
# Vì vậy ngoài hộp, GP không có dữ liệu nào còn PINN vẫn có phương trình. Nếu
# cấu trúc vật lý có giá trị thì chênh lệch phải lộ ra ở phần ngoài hộp.
#   None                       -> toàn không gian (mặc định, các kết quả đã có)
#   {"beta": (0.40, 1.0)}      -> chỉ hỏi ở beta cao, để trống vùng knee
#   {"D": (0.0, 0.65)}         -> chỉ hỏi ở D thấp và trung bình
ORACLE_REGION = None

# ── Ground truth ─────────────────────────────────────────────────────────────
# 25^3 = 15625 lời gọi FEM, khoảng 40 phút ở 0.155 s/lần. Hạ sàn rời rạc hoá
# của delta_H từ 0.062 xuống ~0.038, tức ngưỡng 0.080 không còn nằm sát nhiễu.
# Đổi giá trị này làm cache ground_truth.npz cũ bị xoá và sinh lại, và MỌI
# con số delta_H phải tính lại (không phải huấn luyện lại: checkpoint vẫn dùng
# được). Metrics của BO cũng phải tính lại theo cùng GT, nếu không hai bên
# khác thước đo.
GT_GRID  = 25
GT_SEEDS = 1

# ── Lặp lại ──────────────────────────────────────────────────────────────────
N_SEEDS   = 5
BASE_SEED = 42

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints")
GT_CACHE    = os.path.join(RESULTS_DIR, "ground_truth.npz")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.7)

# ── Tương thích ngược (baselines.py / các script chẩn đoán cũ) ───────────────
N_COLLOCATION = N_PARAM_BATCH * N_COLL_PER_P
ADAM_EPOCHS   = INIT_STEPS
LBFGS_STEPS   = 0        # đã bỏ L-BFGS khỏi vòng trong
FSR_ALERT_THRESH = 0.05

print(f"[config] device={DEVICE}  param_batch={N_PARAM_BATCH}  "
      f"oracle_budget={ORACLE_BUDGET}")