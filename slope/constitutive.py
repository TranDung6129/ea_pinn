"""
slope/constitutive.py — ánh xạ ứng suất theo biến dạng, Hencky + Drucker-Prager.

Đây là BẢN THAM CHIẾU (numpy). Cùng một công thức được viết lại hai lần nữa:
bằng UFL trong oracle.py (FEM) và bằng torch trong PINN. Ba bản phải cho
cùng kết quả; test ở cuối file kiểm các tính chất của bản này, và oracle.py /
PINN sẽ được so với bản này trên cùng các điểm biến dạng.

Mô hình
-------
Lý thuyết biến dạng toàn phần (Hencky): ứng suất là HÀM của biến dạng hiện
tại, không có biến lịch sử. Từ biến dạng eps (3x3, biến dạng phẳng nên
eps_zz = eps_xz = eps_yz = 0):

  1. Ứng suất thử đàn hồi   sig_tr = lam*tr(eps)*I + 2*G*eps
  2. Tiêu chí chảy          f = sqrt(J2) + a*I1 - k      (kéo dương)
     với a, k khớp Mohr-Coulomb cho biến dạng phẳng:
       a = tan(phi) / sqrt(9 + 12 tan^2 phi)
       k = 3c       / sqrt(9 + 12 tan^2 phi)
  3. f_tr <= 0: đàn hồi, sig = sig_tr.
     f_tr  > 0: chiếu MỘT lần về mặt chảy theo quy luật kết hợp
       dgam  = f_tr / (G + 9*K*a^2)
       s     = s_tr * (1 - G*dgam/sqrt(J2_tr))
       I1    = I1_tr - 9*K*a*dgam
     Nếu sqrt(J2_tr) - G*dgam < 0 thì điểm chiếu rơi qua đỉnh nón:
       sig = (k / (3a)) * I      (đỉnh nón, s = 0)

Vì sao chỉ một lần chiếu mà không lặp: với Drucker-Prager kết hợp và đàn hồi
tuyến tính đẳng hướng, phép chiếu về mặt nón có dạng đóng, nên một bước là
nghiệm chính xác của bài toán chiếu, không phải xấp xỉ.

Điều kiện của a > 0 (phi > 0) để đỉnh nón xác định.
"""
import numpy as np


def dp_constants(c, phi_deg):
    t = np.tan(np.radians(phi_deg))
    den = np.sqrt(9.0 + 12.0 * t * t)
    return t / den, 3.0 * c / den          # a, k


def lame(young, nu):
    G = young / (2.0 * (1.0 + nu))
    K = young / (3.0 * (1.0 - 2.0 * nu))
    lam = K - 2.0 * G / 3.0
    return lam, G, K


def stress(eps, young, nu, c, phi_deg, return_info=False):
    """
    eps: (..., 3, 3) biến dạng. Trả về sig (..., 3, 3).
    return_info=True trả thêm dgam (..., ) = độ lớn chảy dẻo, và cờ apex.
    """
    eps = np.asarray(eps, dtype=float)
    lam, G, K = lame(young, nu)
    a, k = dp_constants(c, phi_deg)
    I = np.eye(3)

    tr = np.trace(eps, axis1=-2, axis2=-1)[..., None, None]
    sig_tr = lam * tr * I + 2.0 * G * eps
    I1 = np.trace(sig_tr, axis1=-2, axis2=-1)
    s = sig_tr - (I1 / 3.0)[..., None, None] * I
    sqJ2 = np.sqrt(0.5 * np.sum(s * s, axis=(-2, -1)))

    f = sqJ2 + a * I1 - k
    dgam = np.maximum(f, 0.0) / (G + 9.0 * K * a * a)

    q_new = sqJ2 - G * dgam
    apex = q_new < 0.0
    scale = np.where(sqJ2 > 0.0, np.maximum(q_new, 0.0) / np.where(sqJ2 > 0, sqJ2, 1.0), 0.0)
    I1_new = I1 - 9.0 * K * a * dgam
    sig = s * scale[..., None, None] + (I1_new / 3.0)[..., None, None] * I
    sig = np.where(apex[..., None, None], (k / (3.0 * a)) * I, sig)

    if return_info:
        return sig, dgam, apex
    return sig


def plane_strain(exx, eyy, exy):
    """Dựng tensor 3x3 từ ba thành phần biến dạng phẳng (exy là thành phần tensor)."""
    exx, eyy, exy = np.broadcast_arrays(exx, eyy, exy)
    z = np.zeros_like(exx)
    return np.stack([np.stack([exx, exy, z], -1),
                     np.stack([exy, eyy, z], -1),
                     np.stack([z, z, z], -1)], -2)


# ── Tự kiểm ─────────────────────────────────────────────────────────────────

def _yield(sig, c, phi_deg):
    a, k = dp_constants(c, phi_deg)
    I1 = np.trace(sig, axis1=-2, axis2=-1)
    s = sig - (I1 / 3.0)[..., None, None] * np.eye(3)
    return np.sqrt(0.5 * np.sum(s * s, axis=(-2, -1))) + a * I1 - k


def self_test(n=20000, seed=0):
    """
    Bốn tính chất phải đúng, kiểm trên n biến dạng ngẫu nhiên:
      T1 ứng suất sau chiếu không vượt mặt chảy     max f(sig) <= 1e-8 * k
      T2 điểm chảy dẻo nằm ĐÚNG trên mặt chảy       |f(sig)| nhỏ khi dgam > 0 (trừ đỉnh)
      T3 điểm đàn hồi giữ nguyên ứng suất thử        sig == sig_tr khi dgam == 0
      T4 ánh xạ liên tục: nhiễu nhỏ ở eps cho nhiễu nhỏ ở sig (Lipschitz theo 2G+3K)
    """
    try:
        from slope.config_slope import YOUNG, POISSON, COHESION, PHI_DEG
    except ImportError:
        from config_slope import YOUNG, POISSON, COHESION, PHI_DEG
    rng = np.random.default_rng(seed)
    scale = 3e-4
    e = plane_strain(*(rng.normal(0, scale, (3, n))))
    sig, dgam, apex = stress(e, YOUNG, POISSON, COHESION, PHI_DEG, return_info=True)
    lam, G, K = lame(YOUNG, POISSON)
    a, k = dp_constants(COHESION, PHI_DEG)
    f = _yield(sig, COHESION, PHI_DEG)
    plastic = dgam > 0

    t1 = f.max() <= 1e-8 * k
    t2 = np.abs(f[plastic & ~apex]).max() <= 1e-8 * k if (plastic & ~apex).any() else True
    tr = np.trace(e, axis1=-2, axis2=-1)[..., None, None]
    sig_tr = lam * tr * np.eye(3) + 2 * G * e
    t3 = np.allclose(sig[~plastic], sig_tr[~plastic], rtol=1e-12, atol=1e-9)
    d = plane_strain(*(rng.normal(0, 1e-9, (3, n))))
    ds = np.abs(stress(e + d, YOUNG, POISSON, COHESION, PHI_DEG) - sig).max()
    t4 = ds <= 10 * (2 * G + 3 * K) * 1e-9 * 3

    print(f"  mẫu: {n}, chảy dẻo: {plastic.mean():.1%}, rơi vào đỉnh nón: {apex.mean():.1%}")
    print(f"  T1 max f(sig)/k            = {f.max()/k:+.2e}   {'ĐẠT' if t1 else 'KHÔNG ĐẠT'}")
    fp = np.abs(f[plastic & ~apex]).max() / k if (plastic & ~apex).any() else 0.0
    print(f"  T2 max |f| trên điểm dẻo/k = {fp:.2e}   {'ĐẠT' if t2 else 'KHÔNG ĐẠT'}")
    print(f"  T3 điểm đàn hồi giữ sig_tr               {'ĐẠT' if t3 else 'KHÔNG ĐẠT'}")
    print(f"  T4 max |dsig| với |deps|~1e-9 = {ds:.2e}   {'ĐẠT' if t4 else 'KHÔNG ĐẠT'}")
    ok = t1 and t2 and t3 and t4
    print("  => " + ("tất cả ĐẠT" if ok else "CÓ LỖI, không dùng tiếp"))
    return ok


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print("Tự kiểm ánh xạ Hencky + Drucker-Prager (numpy)")
    sys.exit(0 if self_test() else 1)
