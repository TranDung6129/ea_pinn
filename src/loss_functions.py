"""
src/loss_functions.py
---------------------
L_total = L_PDE + lambda_BC*L_BC + lambda_IC*L_IC + lambda_sup*L_sup

Đã bỏ:
  - loss_event  (rào cản ReLU^2 trên u - u_thr): nó phạt mọi điểm vượt ngưỡng,
    tức là ép mô hình về phía an toàn ở mọi nơi. Bài toán là dựng mặt nghiệm
    E(p)=0, không phải giữ u dưới ngưỡng. Đây là một trong những nguồn làm
    E_pinn lệch về phía an toàn, tức FSR.
  - AdaptiveLossScheduler: lambda_event tăng dần theo |E| là điều chỉnh trọng
    số để chỉ số đẹp lên, không sửa chỗ sai.

Bổ sung:
  - failure_functional dùng LƯỚI CỐ ĐỊNH và có biến thể trơn (smooth max).
    max cứng chỉ cho gradient chảy qua đúng một điểm argmax, nên tín hiệu
    dE/dp trước đây là gradient một điểm. logsumexp cho gradient chảy qua
    mọi điểm gần đỉnh.
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from src.pinn_model import denormalise_params
from src.sampling import eval_grid, boundary_points

# Nhiệt độ cho smooth-max. Càng lớn càng sát max cứng.
SOFTMAX_BETA = getattr(cfg, "SOFTMAX_BETA", 40.0)


def _expand_p(p_hat: torch.Tensor, n: int) -> torch.Tensor:
    return p_hat.unsqueeze(0).expand(n, -1) if p_hat.dim() == 1 else p_hat


# ── PDE residual ──────────────────────────────────────────────────────────────

def pde_residual(model: nn.Module, xyt: torch.Tensor,
                 p_hat: torch.Tensor) -> torch.Tensor:
    """r = u_t - D*(u_xx + u_yy) - alpha*u + beta*u^3"""
    xyt = xyt.clone().requires_grad_(True)
    p_exp = _expand_p(p_hat, xyt.shape[0])

    u = model(xyt, p_exp)
    g = torch.autograd.grad(u.sum(), xyt, create_graph=True)[0]
    u_x, u_y, u_t = g[:, 0], g[:, 1], g[:, 2]
    u_xx = torch.autograd.grad(u_x.sum(), xyt, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y.sum(), xyt, create_graph=True)[0][:, 1]

    alpha, beta, D = denormalise_params(p_exp)
    return u_t - D * (u_xx + u_yy) - alpha * u + beta * u ** 3


def loss_pde(model, xyt, p_hat):
    return (pde_residual(model, xyt, p_hat) ** 2).mean()


# ── Biên và điều kiện ban đầu ─────────────────────────────────────────────────

def loss_bc(model, xyt_bc, p_hat, normals):
    """Neumann: du/dn = 0 trên dOmega."""
    xyt_bc = xyt_bc.clone().requires_grad_(True)
    p_exp = _expand_p(p_hat, xyt_bc.shape[0])
    u = model(xyt_bc, p_exp)
    g = torch.autograd.grad(u.sum(), xyt_bc, create_graph=True)[0]
    return ((g * normals).sum(dim=-1) ** 2).mean()


def loss_ic(model, xy_ic, p_hat):
    """u(x,y,0) = 0.1*sin(pi x)*sin(pi y)"""
    m = xy_ic.shape[0]
    xyt = torch.cat([xy_ic, torch.zeros(m, 1, device=xy_ic.device)], dim=1)
    u_pred = model(xyt, _expand_p(p_hat, m))
    u_true = 0.1 * torch.sin(torch.pi * xy_ic[:, 0]) * torch.sin(torch.pi * xy_ic[:, 1])
    return ((u_pred - u_true) ** 2).mean()


# ── Phiếm hàm thất bại E(u;p) ────────────────────────────────────────────────

def failure_functional(model: nn.Module, p_hat: torch.Tensor,
                       smooth: bool = None, device: str = None) -> torch.Tensor:
    """
    E(u;p) = max_{x,t} u(x,y,t;p) - u_thr, lấy trên lưới đánh giá CỐ ĐỊNH.

    p_hat : (3,) hoặc (B,3)      →  trả về scalar hoặc (B,)
    smooth: mặc định lấy cfg.E_SMOOTH. MỌI nơi (huấn luyện, quét đề xuất, ghi
            E_pinn, đo metrics) phải dùng cùng một định nghĩa. Trước đây huấn
            luyện dùng bản trơn còn đo dùng max cứng, tức mô hình bị chấm
            điểm trên một hàm khác với hàm nó được dạy.
    """
    if smooth is None:
        smooth = getattr(cfg, "E_SMOOTH", True)
    device = device or (p_hat.device if p_hat.is_cuda else cfg.DEVICE)
    grid = eval_grid(str(device))                      # (G,3)
    G = grid.shape[0]
    single = (p_hat.dim() == 1)
    P = p_hat.unsqueeze(0) if single else p_hat        # (B,3)
    B = P.shape[0]

    xyt_rep = grid.unsqueeze(0).expand(B, -1, -1).reshape(B * G, 3)
    p_rep = P.unsqueeze(1).expand(-1, G, -1).reshape(B * G, 3)
    u = model(xyt_rep, p_rep).reshape(B, G)

    if smooth:
        peak = torch.logsumexp(SOFTMAX_BETA * u, dim=1) / SOFTMAX_BETA
    else:
        peak = u.max(dim=1).values

    E = peak - cfg.U_THRESHOLD
    return E.squeeze(0) if single else E


# ── Giám sát từ oracle ────────────────────────────────────────────────────────

def loss_supervised(model, p_batch: torch.Tensor,
                    E_true: torch.Tensor, mode: str = None) -> torch.Tensor:
    """
    Giám sát từ kết quả oracle. Hai chế độ, chọn bằng cfg.SUP_MODE.

    "regress": khớp giá trị E_pinn với E_true, trọng số 1/(|E|+0.3).
        Đây là bài toán khó hơn bài toán thật sự cần: nó bắt mạng 16897 tham
        số tái tạo đúng giá trị E ở mọi nơi. Khi không đủ sức, mạng làm trơn
        trường nghiệm, tức cắt bớt đỉnh, tức E thấp đi có hệ thống.

    "hinge": chỉ đòi E_pinn đúng dấu với một biên độ, không đòi đúng giá trị.
        Thứ cần dựng là mặt E = 0, không phải hàm E. Biên độ lấy bằng
        min(|E_true|, SUP_MARGIN), nên điểm rất sát biên gần như chỉ bị ràng
        buộc về dấu, còn điểm sâu trong miền không bị ép khớp giá trị lớn.
        Loss bằng 0 khi đã đúng dấu và đủ biên độ.
    """
    mode = mode or getattr(cfg, "SUP_MODE", "regress")
    E_pred = failure_functional(model, p_batch)

    if mode == "hinge":
        m = getattr(cfg, "SUP_MARGIN", 0.1)
        margin = torch.clamp(E_true.abs(), max=m)
        viol = torch.clamp(margin - torch.sign(E_true) * E_pred, min=0.0)
        return (viol ** 2).mean()

    w = 1.0 / (E_true.abs() + 0.3)
    return (w * (E_pred - E_true) ** 2).sum() / w.sum()


# ── Tổng hợp ──────────────────────────────────────────────────────────────────

def total_loss(model, xyt_pde, xyt_bc, normals, xy_ic, p_hat):
    l_pde = loss_pde(model, xyt_pde, p_hat)
    l_bc = loss_bc(model, xyt_bc, p_hat, normals)
    l_ic = loss_ic(model, xy_ic, p_hat)
    total = l_pde + cfg.LAMBDA_BC * l_bc + cfg.LAMBDA_IC * l_ic
    return {"total": total, "pde": l_pde.item(),
            "bc": l_bc.item(), "ic": l_ic.item()}


# Tên cũ cho src/baselines.py
sample_bc_points = lambda n, t_end, device: boundary_points(n, device)