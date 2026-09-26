"""
src/sampling.py
---------------
Lấy mẫu điểm collocation và lưới đánh giá E.

Thay thế hoàn toàn src/adaptive_sampling.py cũ (pool 81920 điểm + KD-tree +
argsort mỗi bước). Toàn bộ machinery đó chạy đơn luồng trên CPU trong vòng
huấn luyện và là nguồn nghẽn chính; ngoài ra AdaptiveSampler.update() chỉ
thực sự chạy đúng 1 lần trong cả quá trình (step_count tăng mỗi oracle call,
update_freq=200), nên nó không hề có tác dụng như thiết kế.

Điểm quan trọng: LƯỚI ĐÁNH GIÁ E LÀ CỐ ĐỊNH.
E(u;p) = max_{x,t} u - u_thr. Nếu max lấy trên tập điểm ngẫu nhiên khác nhau
mỗi lần gọi thì E_pinn có nhiễu và lệch hệ thống về phía an toàn (max trên mẫu
hữu hạn luôn nhỏ hơn max thật). Đó là một phần của FSR. Dùng lưới cố định thì
E_pinn là một hàm xác định của (tham số mạng, p).
"""

import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


# ── Lưới đánh giá E cố định ───────────────────────────────────────────────────

_EVAL_NX = getattr(cfg, "EVAL_NX", 12)   # 12x12 điểm không gian
_EVAL_NT = getattr(cfg, "EVAL_NT", 8)    # 8 lát thời gian
_EVAL_CACHE = {}


def eval_grid(device: str = None) -> torch.Tensor:
    """Lưới (x,y,t) cố định dùng để tính E. Cache theo device."""
    device = device or cfg.DEVICE
    if device in _EVAL_CACHE:
        return _EVAL_CACHE[device]
    x = torch.linspace(0.0, 1.0, _EVAL_NX)
    t = torch.linspace(0.0, cfg.T_END, _EVAL_NT)
    XX, YY, TT = torch.meshgrid(x, x, t, indexing="ij")
    grid = torch.stack([XX.reshape(-1), YY.reshape(-1), TT.reshape(-1)], dim=1)
    grid = grid.to(device)
    _EVAL_CACHE[device] = grid
    return grid


def eval_grid_size() -> int:
    return _EVAL_NX * _EVAL_NX * _EVAL_NT


# ── Latin Hypercube trong [0,1]^d ─────────────────────────────────────────────

def lhs(n: int, d: int = 3, device: str = None,
        generator: torch.Generator = None) -> torch.Tensor:
    device = device or cfg.DEVICE
    out = torch.empty(n, d, device=device)
    for j in range(d):
        perm = torch.randperm(n, device=device)
        u = torch.rand(n, device=device)
        out[:, j] = (perm.float() + u) / n
    return out


# ── Vùng cho phép gọi oracle ──────────────────────────────────────────────────
# Phép thử ngoại suy: giới hạn LỜI GỌI ORACLE vào một hộp con của không gian
# tham số, trong khi ràng buộc PDE vẫn phủ toàn bộ không gian (điểm collocation
# vẫn lấy trên cả hộp lớn). Nếu vật lý có tác dụng thì biên ở ngoài hộp vẫn
# phải dựng được, dù chưa từng hỏi oracle ở đó lần nào. GP thì không có gì
# ngoài vùng nó đã hỏi.

def region_bounds(region=None, device: str = None) -> tuple:
    """(lo, hi) dạng tensor (3,) trong toạ độ chuẩn hoá [0,1]^3."""
    device = device or cfg.DEVICE
    region = region if region is not None else getattr(cfg, "ORACLE_REGION", None)
    lo = torch.zeros(3, device=device)
    hi = torch.ones(3, device=device)
    if region:
        for i, k in enumerate(("alpha", "beta", "D")):
            if k in region:
                lo[i] = float(region[k][0])
                hi[i] = float(region[k][1])
    return lo, hi


def region_bounds_np(region=None) -> tuple:
    region = region if region is not None else getattr(cfg, "ORACLE_REGION", None)
    lo = np.zeros(3)
    hi = np.ones(3)
    if region:
        for i, k in enumerate(("alpha", "beta", "D")):
            if k in region:
                lo[i], hi[i] = float(region[k][0]), float(region[k][1])
    return lo, hi


def lhs_region(n: int, device: str = None, region=None) -> torch.Tensor:
    """LHS bên trong vùng cho phép gọi oracle."""
    device = device or cfg.DEVICE
    lo, hi = region_bounds(region, device)
    return lo + (hi - lo) * lhs(n, 3, device)


def clamp_region(p: torch.Tensor, region=None) -> torch.Tensor:
    lo, hi = region_bounds(region, p.device)
    return torch.clamp(p, lo, hi)


def region_is_full(region=None) -> bool:
    lo, hi = region_bounds_np(region)
    return bool(np.allclose(lo, 0.0) and np.allclose(hi, 1.0))


# ── Điểm collocation ──────────────────────────────────────────────────────────

def collocation(n: int, device: str = None) -> torch.Tensor:
    """Điểm (x,y,t) đều trong Omega x [0,T]."""
    device = device or cfg.DEVICE
    pts = torch.rand(n, 3, device=device)
    pts[:, 2] *= cfg.T_END
    return pts


def initial_points(n: int, device: str = None) -> torch.Tensor:
    """Điểm (x,y) cho điều kiện ban đầu."""
    device = device or cfg.DEVICE
    return torch.rand(n, 2, device=device)


def boundary_points(n: int, device: str = None) -> tuple:
    """Điểm trên dOmega x [0,T] kèm pháp tuyến ngoài. (giữ nguyên logic cũ)"""
    device = device or cfg.DEVICE
    n4 = max(1, n // 4)
    r = torch.rand(4, n4, device=device)
    t = torch.rand(4, n4, device=device) * cfg.T_END
    z = torch.zeros(n4, device=device)
    o = torch.ones(n4, device=device)

    pts = torch.cat([
        torch.stack([r[0], z, t[0]], dim=1),      # y=0
        torch.stack([r[1], o, t[1]], dim=1),      # y=1
        torch.stack([z, r[2], t[2]], dim=1),      # x=0
        torch.stack([o, r[3], t[3]], dim=1),      # x=1
    ], dim=0)

    normals = torch.cat([
        torch.stack([z, -o, z], dim=1),
        torch.stack([z,  o, z], dim=1),
        torch.stack([-o, z, z], dim=1),
        torch.stack([o,  z, z], dim=1),
    ], dim=0)
    return pts, normals


# Tên cũ, giữ cho src/baselines.py không phải sửa
_sample_bc = boundary_points