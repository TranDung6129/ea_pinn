"""
slope/energy.py — ứng suất và thế năng biến dạng, bản torch (dùng cho PINN).

Cùng công thức với slope/constitutive.py (numpy) và _sigma_eps trong oracle.py
(UFL). self_test() so bản này với bản numpy.

Thế năng Hencky + Drucker-Prager kết hợp
----------------------------------------
Ứng suất là phép chiếu (theo chuẩn năng lượng C^-1) của ứng suất thử lên tập
ứng suất khả dĩ K = { f(sig) <= 0 }, tập lồi:
    sig(eps) = P_K(C eps)
Hàm
    W(eps) = 1/2 eps:C:eps - 1/2 (sig_tr - sig):C^-1:(sig_tr - sig)
có gradient đúng bằng sig(eps). Lý do: số hạng thứ hai là 1/2 bình phương
khoảng cách (chuẩn C^-1) từ C eps tới K, và gradient của 1/2 dist^2 tới một
tập lồi là (điểm - hình chiếu). Đạo hàm theo eps cho C eps - (C eps - sig) = sig.

Hệ quả cho PINN: cực tiểu hoá năng lượng toàn phần
    Pi(u) = ∫ W(eps(u)) dx - ∫ b·u dx
cho đúng nghiệm của div sig + b = 0 (cùng bài toán FEM giải), mà chỉ cần đạo
hàm BẬC MỘT của u. Dạng mạnh div sig cần đạo hàm bậc hai đi qua chỗ gãy của
phép chiếu (mặt chảy, đỉnh nón), rất nhiễu với autodiff.
"""
import torch


def lame(young, nu):
    G = young / (2.0 * (1.0 + nu))
    K = young / (3.0 * (1.0 - 2.0 * nu))
    return K - 2.0 * G / 3.0, G, K


def dp_constants(c, phi_deg):
    t = torch.tan(torch.deg2rad(torch.as_tensor(phi_deg, dtype=torch.float64)))
    den = torch.sqrt(9.0 + 12.0 * t * t)
    return float(t / den), float(3.0 * c / den)


def material(c, phi_deg, young, nu, srf=1.0):
    """srf chia c và tan(phi), đúng như material() trong oracle.py."""
    import math
    c_r = c / srf
    phi_r = math.degrees(math.atan(math.tan(math.radians(phi_deg)) / srf))
    lam, G, K = lame(young, nu)
    a, k = dp_constants(c_r, phi_r)
    return dict(lam=lam, G=G, K=K, a=a, k=k, c=c_r, phi=phi_r, young=young, nu=nu)


def strain_plane(exx, eyy, exy):
    """(N,) x3 -> (N,3,3), biến dạng phẳng."""
    z = torch.zeros_like(exx)
    return torch.stack([torch.stack([exx, exy, z], -1),
                        torch.stack([exy, eyy, z], -1),
                        torch.stack([z, z, z], -1)], -2)


def stress(eps, m, eps_sq=1e-12):
    """eps (N,3,3) -> sig (N,3,3), dgam (N,), cờ đỉnh nón (N,)."""
    lam, G, K, a, k = m["lam"], m["G"], m["K"], m["a"], m["k"]
    I = torch.eye(3, dtype=eps.dtype, device=eps.device)
    tr = eps.diagonal(dim1=-2, dim2=-1).sum(-1)
    sig_tr = lam * tr[:, None, None] * I + 2.0 * G * eps
    I1 = sig_tr.diagonal(dim1=-2, dim2=-1).sum(-1)
    s = sig_tr - (I1 / 3.0)[:, None, None] * I
    sqJ2 = torch.sqrt(0.5 * (s * s).sum((-2, -1)) + eps_sq)
    f = sqJ2 + a * I1 - k
    dgam = torch.relu(f) / (G + 9.0 * K * a * a)
    q_new = sqJ2 - G * dgam
    apex = q_new < 0.0
    cone = s * (q_new / sqJ2)[:, None, None] + ((I1 - 9.0 * K * a * dgam) / 3.0)[:, None, None] * I
    kap = k / (3.0 * a)                      # a, k có thể là số hoặc tensor (N,) theo điểm
    if torch.is_tensor(kap) and kap.dim() == 1:
        kap = kap[:, None, None]
    sig = torch.where(apex[:, None, None], kap * I, cone)
    return sig, sig_tr, dgam, apex


def energy_density(eps, m):
    """W(eps) (N,). Gradient theo eps bằng sig(eps)."""
    lam, G = m["lam"], m["G"]
    sig, sig_tr, _, _ = stress(eps, m)
    tr_e = eps.diagonal(dim1=-2, dim2=-1).sum(-1)
    W_el = 0.5 * (sig_tr * eps).sum((-2, -1))
    d = sig_tr - sig
    tr_d = d.diagonal(dim1=-2, dim2=-1).sum(-1)
    I = torch.eye(3, dtype=eps.dtype, device=eps.device)
    Cinv_d = (d - (lam / (3.0 * lam + 2.0 * G)) * tr_d[:, None, None] * I) / (2.0 * G)
    return W_el - 0.5 * (d * Cinv_d).sum((-2, -1))


# ── Tự kiểm ─────────────────────────────────────────────────────────────────

def self_test(n=4000, seed=0):
    """
    E1 sig (torch) == sig (numpy, slope/constitutive.py)          <= 1e-9 k
    E2 dW/deps (autodiff) == sig trên ba thành phần phẳng         <= 1e-6 k
       (dW/d exy của tensor đối xứng = 2 sig_xy)
    """
    import numpy as np
    from slope import config_slope as cs, constitutive as cn
    m = material(cs.COHESION, cs.PHI_DEG, cs.YOUNG, cs.POISSON, srf=1.3)
    g = torch.Generator().manual_seed(seed)
    e3 = torch.randn(3, n, generator=g, dtype=torch.float64) * 3e-4
    exx, eyy, exy = [x.clone().requires_grad_(True) for x in e3]
    eps = strain_plane(exx, eyy, exy)
    sig, _, dgam, apex = stress(eps, m)
    s_np = cn.stress(cn.plane_strain(*e3.numpy()), m["young"], m["nu"], m["c"], m["phi"])
    e1 = np.abs(sig.detach().numpy() - s_np).max() / m["k"]

    W = energy_density(eps, m).sum()
    gx, gy, gxy = torch.autograd.grad(W, [exx, eyy, exy])
    e2 = max((gx - sig[:, 0, 0]).abs().max().item(),
             (gy - sig[:, 1, 1]).abs().max().item(),
             (gxy - 2 * sig[:, 0, 1]).abs().max().item()) / m["k"]
    print(f"  mẫu {n}: dẻo {(dgam > 0).float().mean():.1%}, đỉnh nón {apex.float().mean():.1%}")
    print(f"  E1 torch so với numpy: max lệch / k = {e1:.2e}  {'ĐẠT' if e1 < 1e-9 else 'KHÔNG ĐẠT'}")
    print(f"  E2 dW/deps so với sig: max lệch / k = {e2:.2e}  {'ĐẠT' if e2 < 1e-6 else 'KHÔNG ĐẠT'}")
    return e1 < 1e-9 and e2 < 1e-6


if __name__ == "__main__":
    import sys
    print("Tự kiểm thế năng Hencky + Drucker-Prager (torch)")
    sys.exit(0 if self_test() else 1)
