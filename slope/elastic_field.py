"""
slope/elastic_field.py — dựng lại CHÍNH XÁC trường FEM P2 từ file .npz của oracle,
để dùng lời giải đàn hồi FEM làm phần nền cho PINN dạng phần dư.

    u(x; p) = u_el(x; kh) + u_res(x; p)
    u_el(x; kh) = u_el0(x) + kh * (u_el(kh1) - u_el0)(x) / kh1     (đàn hồi tuyến tính)

File oracle có đỉnh P1 (geom_x, cells) và toạ độ bậc tự do P2 (coords, u). Mỗi
tam giác có 3 bậc tự do ở đỉnh và 3 ở trung điểm cạnh; hàm dạng P2 theo toạ độ
trọng tâm L: đỉnh N_i = L_i(2L_i - 1), cạnh (i,j) N = 4 L_i L_j. Đây là đúng
nghiệm FEM, không phải nội suy xấp xỉ.

Tự kiểm (chỉ cần numpy, scipy, matplotlib):
    python -m slope.elastic_field
"""
import json, os, sys
import numpy as np
import matplotlib.tri as mtri

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_slope")


class P2Field:
    def __init__(self, npz):
        d = np.load(npz) if isinstance(npz, str) else npz
        X, T = d["geom_x"], d["cells"].astype(int)
        C, self.U = d["coords"], d["u"]
        key = {tuple(np.round(p, 8)): i for i, p in enumerate(C)}
        find = lambda p: key[tuple(np.round(p, 8))]
        self.dofs = np.array([[find(X[a]), find(X[b]), find(X[c]),
                               find(0.5 * (X[a] + X[b])), find(0.5 * (X[b] + X[c])),
                               find(0.5 * (X[c] + X[a]))] for a, b, c in T])
        self.X, self.T = X, T
        self.tri = mtri.Triangulation(X[:, 0], X[:, 1], T)
        self.finder = self.tri.get_trifinder()
        # toạ độ trọng tâm: L = A^-1 [x - x3; y - y3]
        P0, P1, P2 = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
        A = np.stack([P0 - P2, P1 - P2], -1)                     # (nt, 2, 2)
        self.Ainv, self.P2 = np.linalg.inv(A), P2

    def _locate(self, xy):
        t = self.finder(xy[:, 0], xy[:, 1])
        bad = t < 0
        if bad.any():   # điểm nằm đúng trên biên ngoài: lấy tam giác có tâm gần nhất
            cen = self.X[self.T].mean(1)
            for k in np.where(bad)[0]:
                t[k] = np.argmin(np.linalg.norm(cen - xy[k], axis=1))
        return t

    def eval(self, xy, grad=False):
        """u (N,2); nếu grad=True trả thêm G (N,2,2) với G[:, i, j] = du_i/dx_j."""
        t = self._locate(xy)
        l12 = np.einsum("nij,nj->ni", self.Ainv[t], xy - self.P2[t])
        L = np.column_stack([l12, 1 - l12.sum(1)])
        N = np.column_stack([L[:, 0] * (2 * L[:, 0] - 1), L[:, 1] * (2 * L[:, 1] - 1),
                             L[:, 2] * (2 * L[:, 2] - 1), 4 * L[:, 0] * L[:, 1],
                             4 * L[:, 1] * L[:, 2], 4 * L[:, 2] * L[:, 0]])
        Ue = self.U[self.dofs[t]]                                # (N, 6, 2)
        u = np.einsum("nk,nkc->nc", N, Ue)
        if not grad:
            return u
        dL = np.concatenate([self.Ainv[t], -self.Ainv[t].sum(1, keepdims=True)], 1)  # (N,3,2)
        dN = np.stack([(4 * L[:, 0:1] - 1) * dL[:, 0], (4 * L[:, 1:2] - 1) * dL[:, 1],
                       (4 * L[:, 2:3] - 1) * dL[:, 2],
                       4 * (L[:, 0:1] * dL[:, 1] + L[:, 1:2] * dL[:, 0]),
                       4 * (L[:, 1:2] * dL[:, 2] + L[:, 2:3] * dL[:, 1]),
                       4 * (L[:, 2:3] * dL[:, 0] + L[:, 0:1] * dL[:, 2])], 1)       # (N,6,2)
        G = np.einsum("nkc,nkj->ncj", Ue, dN)
        return u, G


class ElasticBase:
    """u_el(x; kh) tuyến tính theo kh, dựng từ hai lời giải FEM đàn hồi."""
    def __init__(self, f0="elastic_kh0.npz", f1="elastic_kh01.npz"):
        d1 = np.load(os.path.join(RES, f1))
        self.kh1 = json.loads(str(d1["params"]))["kh"]
        self.F0 = P2Field(os.path.join(RES, f0))
        self.F1 = P2Field(d1)

    def at(self, xy, grad=True):
        """Trả (u0, du/dkh) và nếu grad thì (G0, dG/dkh); đều theo mét."""
        if grad:
            u0, G0 = self.F0.eval(xy, True)
            u1, G1 = self.F1.eval(xy, True)
            return u0, (u1 - u0) / self.kh1, G0, (G1 - G0) / self.kh1
        u0, u1 = self.F0.eval(xy), self.F1.eval(xy)
        return u0, (u1 - u0) / self.kh1


def self_test():
    from slope import config_slope as cs
    from slope.pinn import quadrature
    d = np.load(os.path.join(RES, "elastic_kh0.npz"))
    F = P2Field(d)
    e1 = np.abs(F.eval(d["coords"]) - d["u"]).max() / np.abs(d["u"]).max()
    print(f"  T1 tái tạo u tại bậc tự do P2: lệch tương đối {e1:.1e}  {'ĐẠT' if e1 < 1e-10 else 'KHÔNG ĐẠT'}")
    uc = F.eval((d["crest"] + np.array([1e-6, -1e-6]))[None])[0]
    e2 = np.abs(uc - d["u_crest"][:2]).max() / np.abs(d["u_crest"]).max()
    print(f"  T2 u tại mép đỉnh khớp oracle: lệch {e2:.1e}  {'ĐẠT' if e2 < 1e-6 else 'KHÔNG ĐẠT'}")
    # T3: lời giải đàn hồi Galerkin thoả năng lượng biến dạng = 1/2 công ngoại lực
    q, w, _ = quadrature(h=0.5)
    u, G = F.eval(q, grad=True)
    lam = cs.YOUNG * cs.POISSON / ((1 + cs.POISSON) * (1 - 2 * cs.POISSON))
    mu = cs.YOUNG / (2 * (1 + cs.POISSON))
    e = 0.5 * (G + G.transpose(0, 2, 1))
    tr = e[:, 0, 0] + e[:, 1, 1]
    Wel = 0.5 * lam * tr ** 2 + mu * (e ** 2).sum((1, 2))
    Ustrain, Wext = (w * Wel).sum(), (w * (-cs.GAMMA) * u[:, 1]).sum()
    e3 = abs(Ustrain - 0.5 * Wext) / abs(0.5 * Wext)
    print(f"  T3 năng lượng biến dạng / (1/2 công ngoại lực) lệch {e3:.1e}"
          f"  {'ĐẠT' if e3 < 1e-3 else 'KHÔNG ĐẠT'} (ngưỡng 1e-3, cầu phương PINN không trùng lưới FEM)")
    return e1 < 1e-10 and e2 < 1e-6 and e3 < 1e-3


if __name__ == "__main__":
    print("Tự kiểm dựng lại trường FEM P2")
    sys.exit(0 if self_test() else 1)