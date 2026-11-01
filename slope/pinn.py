"""
slope/pinn.py — PINN cho mái dốc đàn dẻo, cổng 1 (một bộ tham số cố định).

Chạy trong pinn_venv (torch, numpy, scipy, matplotlib; KHÔNG cần dolfinx):

    python -m slope.pinn --ref results_slope/gate1_srf13_fine.npz --srf 1.3
    python -m slope.pinn --ref ... --srf 1.3 --adam 20000 --lbfgs 2000 --seed 43

Cách huấn luyện: cực tiểu hoá năng lượng toàn phần (phương pháp năng lượng),
xem slope/energy.py. Không dùng một điểm dữ liệu FEM nào khi huấn luyện:
nghiệm FEM (--ref) chỉ dùng để ĐÁNH GIÁ sau cùng, qua slope/compare.py.

Điều kiện biên đặt cứng qua dạng nghiệm thử:
    u_x = U0 * B_x(x,y) * N_x(x,y),   B_x = 4 xh (1 - xh) yh   (0 ở hai bên và đáy)
    u_y = U0 * B_y(x,y) * N_y(x,y),   B_y = yh                 (0 ở đáy)
với xh = x/W, yh = y/Y_top. Mặt trên tự do: điều kiện tự nhiên, năng lượng
tự thoả, không cần số hạng phạt.

Tích phân năng lượng: tam giác hoá miền bằng lưới điểm đều + làm mịn quanh
chân và mép đỉnh dốc (lưới riêng của PINN, không dùng lưới FEM), quy tắc 3
điểm trung điểm cạnh (chính xác bậc 2) trên mỗi tam giác.

Đầu ra: results_slope/pinn_<tên>.npz, cùng khoá với file của oracle, nên
chạy thẳng được slope.compare.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slope import config_slope as cs
from slope import energy as en

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results_slope")
DTYPE = torch.float64      # bài toán có u cỡ 1e-3 m và ứng suất cỡ 1e2 kPa;
                           # float32 làm nhiễu phần năng lượng dẻo đáng kể


# ── Hình học (khớp oracle.slope_points) ─────────────────────────────────────

def geometry():
    x_crest = cs.L_CREST
    x_toe = x_crest + cs.SLOPE_RUN
    W = x_toe + cs.L_TOE
    y_toe, y_top = cs.H_FOUND, cs.H_FOUND + cs.H_SLOPE
    poly = np.array([(0, 0), (W, 0), (W, y_toe), (x_toe, y_toe),
                     (x_crest, y_top), (0, y_top)], dtype=float)
    return poly, np.array([x_crest, y_top]), np.array([x_toe, y_toe]), W, y_top


def _inside(pts, poly):
    from matplotlib.path import Path
    return Path(poly).contains_points(pts, radius=1e-9)


def quadrature(h=0.25, h_fine=0.06, r_fine=1.5):
    """Điểm cầu phương (N,2) và trọng số (N,) phủ đúng miền đa giác."""
    from scipy.spatial import Delaunay
    poly, crest, toe, W, Yt = geometry()
    xs = np.arange(0, W + 1e-9, h)
    ys = np.arange(0, Yt + 1e-9, h)
    P = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    # điểm trên các cạnh để tam giác hoá khớp biên (nhất là mặt dốc xiên)
    edges = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        n = max(int(np.ceil(np.linalg.norm(b - a) / h)), 1)
        edges.append(a + (b - a) * np.linspace(0, 1, n + 1)[:, None])
    # làm mịn quanh chân dốc (điểm kỳ dị) và mép đỉnh
    fine = []
    for c in (toe, crest):
        g = np.arange(-r_fine, r_fine + 1e-9, h_fine)
        Q = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) + c
        fine.append(Q[np.linalg.norm(Q - c, axis=1) <= r_fine])
        for i in range(len(poly)):                 # điểm biên mịn quanh góc
            a, b = poly[i], poly[(i + 1) % len(poly)]
            n = max(int(np.ceil(np.linalg.norm(b - a) / h_fine)), 1)
            E = a + (b - a) * np.linspace(0, 1, n + 1)[:, None]
            fine.append(E[np.linalg.norm(E - c, axis=1) <= r_fine])
    P = np.vstack([P] + edges + fine)
    P = P[_inside(P, poly)]
    P = np.unique(np.round(P, 9), axis=0)
    tri = Delaunay(P)
    T = tri.simplices
    cen = P[T].mean(1)
    T = T[_inside(cen, poly)]                       # bỏ tam giác nằm ở chỗ lõm
    d1, d2 = P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]
    A = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    mids = np.concatenate([0.5 * (P[T[:, 0]] + P[T[:, 1]]),
                           0.5 * (P[T[:, 1]] + P[T[:, 2]]),
                           0.5 * (P[T[:, 2]] + P[T[:, 0]])])
    w = np.concatenate([A / 3, A / 3, A / 3])
    area_exact = cs.L_CREST * Yt + cs.SLOPE_RUN * (cs.H_FOUND + 0.5 * cs.H_SLOPE) \
        + cs.L_TOE * cs.H_FOUND
    return mids, w, dict(n_tri=len(T), area=A.sum(), area_exact=area_exact)


# ── Mạng ────────────────────────────────────────────────────────────────────

class SlopePINN(nn.Module):
    def __init__(self, width=64, depth=5, n_fourier=0, fourier_scale=1.0, seed=0):
        super().__init__()
        _, _, _, self.W, self.Yt = geometry()
        self.U0 = 0.01                                       # thang chuyển vị (m)
        g = torch.Generator().manual_seed(seed)
        self.n_fourier = n_fourier
        if n_fourier:
            self.register_buffer("B", torch.randn(n_fourier, 2, generator=g,
                                                  dtype=DTYPE) * fourier_scale)
        d_in = 2 + 2 * n_fourier
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers).to(DTYPE)
        for mod in self.net:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, xy):
        xh = xy[:, 0:1] / self.W
        yh = xy[:, 1:2] / self.Yt
        z = torch.cat([2 * xh - 1, 2 * yh - 1], 1)
        if self.n_fourier:
            proj = 2 * torch.pi * z @ self.B.T
            z = torch.cat([z, torch.sin(proj), torch.cos(proj)], 1)
        N = self.net(z)
        ux = self.U0 * 4 * xh * (1 - xh) * yh * N[:, 0:1]
        uy = self.U0 * yh * N[:, 1:2]
        return torch.cat([ux, uy], 1)


def strain(model, xy):
    xy = xy.clone().requires_grad_(True)
    u = model(xy)
    gx = torch.autograd.grad(u[:, 0].sum(), xy, create_graph=True)[0]
    gy = torch.autograd.grad(u[:, 1].sum(), xy, create_graph=True)[0]
    exx, eyy, exy = gx[:, 0], gy[:, 1], 0.5 * (gx[:, 1] + gy[:, 0])
    return en.strain_plane(exx, eyy, exy), u


def total_energy(model, xy, w, mat, gamma):
    eps, u = strain(model, xy)
    Wd = en.energy_density(eps, mat)
    return (w * Wd).sum() + gamma * (w * u[:, 1]).sum()    # b = (0, -gamma)


# ── Huấn luyện ──────────────────────────────────────────────────────────────

def train(srf=1.0, adam=15000, lbfgs=1500, lr=1e-3, width=64, depth=5,
          n_fourier=0, h=0.25, seed=42, log_every=1000, device="cpu", adam_eps=1e-8,
          h_fine=0.06, r_fine=1.5):
    torch.manual_seed(seed)
    mat = en.material(cs.COHESION, cs.PHI_DEG, cs.YOUNG, cs.POISSON, srf=srf)
    q, w, info = quadrature(h=h, h_fine=h_fine, r_fine=r_fine)
    print(f"  cầu phương: {len(q)} điểm, {info['n_tri']} tam giác, "
          f"diện tích {info['area']:.3f} (đúng {info['area_exact']:.3f})")
    xy = torch.tensor(q, dtype=DTYPE, device=device)
    wt = torch.tensor(w, dtype=DTYPE, device=device)
    model = SlopePINN(width, depth, n_fourier, seed=seed).to(device)
    # thang năng lượng để loss cỡ O(1): trọng lực * chiều cao * U0 * diện tích
    scale = cs.GAMMA * model.Yt * model.U0 * info["area"]
    hist, t0 = [], time.time()

    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=adam_eps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam, 1),
                                                       eta_min=lr * 0.02)
    for it in range(adam):
        opt.zero_grad()
        loss = total_energy(model, xy, wt, mat, cs.GAMMA) / scale
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == adam - 1:
            hist.append((it, loss.item()))
            print(f"    adam {it:6d}  Pi/scale = {loss.item():+.8f}  {time.time()-t0:6.0f} s")

    if lbfgs:
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs,
                                history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            l = total_energy(model, xy, wt, mat, cs.GAMMA) / scale
            l.backward()
            return l
        loss = opt.step(closure)
        hist.append((adam + lbfgs, float(loss)))
        print(f"    lbfgs        Pi/scale = {float(loss):+.8f}  {time.time()-t0:6.0f} s")
    return model, mat, dict(hist=hist, seconds=time.time() - t0, quad=info)


# ── Xuất kết quả theo định dạng oracle ──────────────────────────────────────

def export(model, mat, ref, meta, srf):
    _, crest, toe, _, _ = geometry()
    X = torch.tensor(ref["coords"], dtype=DTYPE)
    with torch.no_grad():
        U = model(X).numpy()
    C = torch.tensor(np.vstack([ref["centroids"],
                                [crest + np.array([1e-6, -1e-6])]]), dtype=DTYPE)
    eps, uc = strain(model, C)
    _, _, dgam, apex = en.stress(eps.detach(), mat)
    u_crest = uc[-1].detach().numpy()
    mag = float(np.linalg.norm(u_crest))
    return dict(converged=True, u=U, coords=ref["coords"],
                centroids=ref["centroids"], dgam=dgam[:-1].numpy(),
                apex=apex[:-1].numpy().astype(float),
                u_crest=u_crest, u_crest_mag=mag, E=mag / cs.U_ALARM - 1.0,
                crest=crest, toe=toe, seconds=meta["seconds"],
                params=json.dumps(dict(srf=srf, **{k: v for k, v in meta.items()
                                                   if k in ("width", "depth", "seed",
                                                            "n_fourier", "adam", "lbfgs")})),
                hist=np.array(meta["hist"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="npz của oracle, chỉ dùng để đánh giá")
    ap.add_argument("--srf", type=float, default=1.3)
    ap.add_argument("--adam", type=int, default=15000)
    ap.add_argument("--lbfgs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--fourier", type=int, default=0)
    ap.add_argument("--h", type=float, default=0.25)
    ap.add_argument("--h-fine", type=float, default=0.06)
    ap.add_argument("--r-fine", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    global DTYPE
    if a.float32:
        DTYPE = torch.float32

    from slope import compare
    ref = compare.load(a.ref)
    srf_ref = json.loads(str(ref["params"]))["srf"]
    if abs(srf_ref - a.srf) > 1e-9:
        sys.exit(f"--srf {a.srf} khác srf của tham chiếu ({srf_ref})")

    print(f"PINN năng lượng  srf={a.srf}  seed={a.seed}  thiết bị={a.device}")
    model, mat, meta = train(a.srf, a.adam, a.lbfgs, a.lr, a.width, a.depth,
                             a.fourier, a.h, a.seed, device=a.device, adam_eps=a.adam_eps,
                             h_fine=a.h_fine, r_fine=a.r_fine)
    model = model.cpu()
    meta.update(width=a.width, depth=a.depth, seed=a.seed, n_fourier=a.fourier,
                adam=a.adam, lbfgs=a.lbfgs)
    out = export(model, mat, ref, meta, a.srf)

    name = a.name or f"pinn_srf{a.srf:g}_seed{a.seed}"
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez_compressed(path, **out)
    torch.save(model.state_dict(), path.replace(".npz", ".pt"))
    print(f"  đã lưu {path}")

    m = compare.metrics(ref, out)
    print(f"  M1 L2 tương đối của u      = {m['M1_L2_rel']:.4f}")
    print(f"  M2 lệch |u_đỉnh| tương đối = {m['M2_crest_rel']:.4f}"
          f"   ({m['crest_ref_mm']:.3f} FEM, {m['crest_oth_mm']:.3f} PINN mm)")
    print(f"  M3 IoU vùng dẻo            = {m['M3_iou']:.3f}"
          f"   (dẻo FEM {m['plastic_frac_ref']:.1%}, PINN {m['plastic_frac_oth']:.1%})")
    ok = True
    for key, (op, t) in compare.GATE1.items():
        good = m[key] <= t if op == "<=" else m[key] >= t
        ok &= good
        print(f"  {key:14s} {op} {t:<5} {'ĐẠT' if good else 'KHÔNG ĐẠT'}")
    print("  => cổng 1 " + ("ĐẠT" if ok else "KHÔNG ĐẠT"))


if __name__ == "__main__":
    main()
