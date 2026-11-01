"""
slope/pinn_param.py — PINN tham số cho cổng 2: u(x, y; c, phi, kh) trên hộp tham số.

Chạy trong pinn_venv:
    python -m slope.pinn_param                          # mặc định: 20000 bước, P=8, nsub=2400
    python -m slope.pinn_param --steps 1500 --name check   # phép thử ngắn trước khi chạy dài

Huấn luyện: cực tiểu hoá TRUNG BÌNH năng lượng toàn phần theo p trong hộp.
Mỗi bước lấy P bộ tham số đều trong hộp; với mỗi bộ, lấy ngẫu nhiên nsub điểm
cầu phương (trọng số nhân N/nsub, nên ước lượng năng lượng không chệch).
Không dùng dữ liệu FEM khi huấn luyện.

    Pi(u; p) = ∫ W(eps(u); c, phi) dx - ∫ b·u dx,   b = (kh*gamma, -gamma)

Đánh giá (sau huấn luyện) trên 48 điểm quét FEM trong hộp (scan_gate2.npz),
định nghĩa E theo results_slope/gate2_def.json (chuyển vị dư tại mép đỉnh).

Chế độ phần dư (--residual, thêm 28/09 sau chẩn đoán lần chạy gate2_seed42):
    u(x; p) = u_el(x; kh) + u_res(x; p)
u_el là lời giải FEM đàn hồi (tuyến tính theo kh, dựng lại chính xác từ
elastic_kh0.npz và elastic_kh01.npz, xem slope/elastic_field.py). Mạng chỉ học
u_res, với thang U0 = --u0 (mặc định 1 mm thay cho 10 mm). Năng lượng vẫn tính
trên chuyển vị tổng, nên bài toán cực tiểu không đổi; chỉ đổi thứ mạng phải
biểu diễn. E của PINN khi đó bằng |u_res(đỉnh)| / U_ALARM - 1.

Ngưỡng cổng 2, chốt 27/09 trước khi huấn luyện:
  G1  trung vị M1 (L2 tương đối của u) trên 48 điểm            <= 0.05
  G2  trung vị M3 (IoU vùng dẻo, dgam > 5e-4) trên các điểm mà
      FEM có >= 1% phần tử dẻo                                   >= 0.60
  G3  tỉ lệ đúng dấu E (xem SỬA 28/09 bên dưới), trên điểm kiểm tra
      ngoài dải nhiễu FEM |E_FEM| < 0.15                         >= 0.90
Hiệu chỉnh mức: 8 điểm hiệu chỉnh chọn ngẫu nhiên (seed 0) trong 48 điểm;
độ lệch b = trung bình (u_đỉnh FEM - u_đỉnh PINN) trên 8 điểm đó, cộng vào
u_đỉnh PINN ở 40 điểm kiểm tra. Tỉ lệ đúng dấu TRƯỚC hiệu chỉnh cũng báo cáo.

SỬA 28/09, trước lần chạy phần dư đầu tiên (chỉ mới thấy lần chạy thử 30 bước):
G3 dùng tỉ lệ đúng dấu KHÔNG hiệu chỉnh (E trực tiếp của PINN). Cả hai cách hiệu
chỉnh (độ lệch hằng, hệ số) chỉ được báo cáo như phân tích phụ. Lý do: không có
căn cứ chốt trước cách hiệu chỉnh nào đúng cho dạng phần dư, và chọn cách hiệu
chỉnh sau khi thấy kết quả là tự chấm điểm. Neo bằng FEM thuộc về vòng lặp chủ
động (cổng 3), nơi lời gọi FEM đặt gần biên chứ không phải 8 điểm ngẫu nhiên.
Ghi chú về hiệu chỉnh hệ số (phân tích phụ): Chẩn đoán gate2_seed42 cho
thấy sai số u_đỉnh không phải hằng số (trung bình 0.07 mm, độ lệch chuẩn 0.17 mm)
và PINN chỉ tạo ra khoảng 1/3 chuyển vị dư ở góc yếu. Với phần dư r = u_đỉnh - u_el,
hệ số s = sum(r_FEM · r_PINN) / sum(|r_PINN|^2) trên 8 điểm hiệu chỉnh, rồi
E = |s r_PINN| / U_ALARM - 1. Điểm đàn hồi (r = 0) giữ nguyên r = 0; độ lệch hằng
thì đẩy chúng thành E dương. Kết quả theo độ lệch hằng vẫn được in để so.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slope import config_slope as cs
from slope import energy as en
from slope.pinn import geometry, quadrature

DTYPE = torch.float64
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_slope")
DGAM_THR, NOISE_E, N_CAL = 5e-4, 0.15, 8


# ── Mạng ────────────────────────────────────────────────────────────────────

class ParamPINN(nn.Module):
    def __init__(self, box, width=64, depth=5, u0=0.01):
        super().__init__()
        _, _, _, self.W, self.Yt = geometry()
        self.U0 = u0
        lo = torch.tensor([box["c"][0], box["phi"][0], box["kh"][0]], dtype=DTYPE)
        hi = torch.tensor([box["c"][1], box["phi"][1], box["kh"][1]], dtype=DTYPE)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        layers, d = [], 5
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers).to(DTYPE)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy, p):
        """xy (N,2) mét, p (N,3) = (c kPa, phi độ, kh) vật lý."""
        xh, yh = xy[:, 0:1] / self.W, xy[:, 1:2] / self.Yt
        ph = 2 * (p - self.lo) / (self.hi - self.lo) - 1
        N = self.net(torch.cat([2 * xh - 1, 2 * yh - 1, ph], 1))
        return torch.cat([self.U0 * 4 * xh * (1 - xh) * yh * N[:, 0:1],
                          self.U0 * yh * N[:, 1:2]], 1)


def material(p):
    """Hằng số Drucker-Prager theo từng điểm từ p (N,3). Đàn hồi cố định."""
    lam, G, K = en.lame(cs.YOUNG, cs.POISSON)
    t = torch.tan(torch.deg2rad(p[:, 1]))
    den = torch.sqrt(9 + 12 * t * t)
    return dict(lam=lam, G=G, K=K, a=t / den, k=3 * p[:, 0] / den)


def strain(model, xy, p, create_graph=True, base=None):
    """base = (u_b (N,2), G_b (N,2,2)): trường nền cộng vào (chế độ phần dư)."""
    xy = xy.clone().requires_grad_(True)
    u = model(xy, p)
    gx = torch.autograd.grad(u[:, 0].sum(), xy, create_graph=create_graph, retain_graph=True)[0]
    gy = torch.autograd.grad(u[:, 1].sum(), xy, create_graph=create_graph)[0]
    uxx, uxy, uyx, uyy = gx[:, 0], gx[:, 1], gy[:, 0], gy[:, 1]
    if base is not None:
        ub, Gb = base
        u = u + ub
        uxx, uxy, uyx, uyy = uxx + Gb[:, 0, 0], uxy + Gb[:, 0, 1], uyx + Gb[:, 1, 0], uyy + Gb[:, 1, 1]
    return en.strain_plane(uxx, uyy, 0.5 * (uxy + uyx)), u


class Base:
    """u_el và grad u_el tại một tập điểm cố định, tuyến tính theo kh."""
    def __init__(self, eb, xy_np, device, grad=True):
        t = lambda a: torch.tensor(a, dtype=DTYPE, device=device)
        r = eb.at(xy_np, grad=grad)
        self.u0, self.du = t(r[0]), t(r[1])
        self.G0, self.dG = (t(r[2]), t(r[3])) if grad else (None, None)

    def at(self, kh, idx=None):
        sel = (lambda a: a) if idx is None else (lambda a: a[idx])
        k = kh[:, None]
        u = sel(self.u0) + k * sel(self.du)
        G = None if self.G0 is None else sel(self.G0) + k[:, :, None] * sel(self.dG)
        return u, G


# ── Huấn luyện ──────────────────────────────────────────────────────────────

def train(box, steps, P, nsub, lr, width, depth, h, seed, device, log_every=1000,
          monitor=None, monitor_every=0, eb=None, u0=0.01):
    torch.manual_seed(seed)
    q, w, info = quadrature(h=h)
    Q = torch.tensor(q, dtype=DTYPE, device=device)
    Wq = torch.tensor(w, dtype=DTYPE, device=device)
    Nq = len(q)
    nsub = min(nsub, Nq)
    model = ParamPINN(box, width, depth, u0).to(device)
    if eb is not None:      # phần dư: bắt đầu đúng tại lời giải đàn hồi (u_res = 0)
        nn.init.zeros_(model.net[-1].weight)
        nn.init.zeros_(model.net[-1].bias)
    base = Base(eb, q, device) if eb is not None else None
    scale = cs.GAMMA * model.Yt * 0.01 * info["area"]      # giữ thang cũ để log so được
    lo, hi = model.lo, model.hi
    print(f"  cầu phương {Nq} điểm, mỗi bước {P} bộ tham số x {nsub} điểm = {P*nsub}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    t0 = time.time()
    for it in range(steps):
        pp = lo + (hi - lo) * torch.rand(P, 3, dtype=DTYPE, device=device)
        idx = torch.stack([torch.randperm(Nq, device=device)[:nsub] for _ in range(P)])
        xy = Q[idx].reshape(-1, 2)
        wt = (Wq[idx] * (Nq / nsub)).reshape(-1)
        pr = pp.repeat_interleave(nsub, 0)
        bb = None if base is None else base.at(pr[:, 2], idx.reshape(-1))
        eps, u = strain(model, xy, pr, base=bb)
        Wd = en.energy_density(eps, material(pr))
        Pi = wt * Wd + cs.GAMMA * wt * u[:, 1] - pr[:, 2] * cs.GAMMA * wt * u[:, 0]
        loss = Pi.sum() / P / scale
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == steps - 1:
            print(f"    bước {it:6d}  Pi/scale (lô) = {loss.item():+.8f}  {time.time()-t0:6.0f} s")
        if monitor and monitor_every and it and it % monitor_every == 0:
            monitor(model, it)
    return model, time.time() - t0


# ── Đánh giá ────────────────────────────────────────────────────────────────

def E_of(u_crest_m, kh, gdef):
    u_el = np.array(gdef["u_el0_m"]) + kh[:, None] * np.array(gdef["du_el_dkh_m"])
    return np.linalg.norm(u_crest_m - u_el, axis=1) * 1e3 / gdef["U_ALARM_mm"] - 1.0


def crest_pred(model, P, device, eb=None):
    _, crest, _, _, _ = geometry()
    xyn = np.tile(crest + np.array([1e-6, -1e-6]), (len(P), 1))
    xy = torch.tensor(xyn, dtype=DTYPE, device=device)
    Pt = torch.tensor(P, dtype=DTYPE, device=device)
    with torch.no_grad():
        u = model(xy, Pt)
        if eb is not None:
            u = u + Base(eb, xyn[:1], device, grad=False).at(Pt[:, 2], torch.zeros(len(P), dtype=torch.long, device=device))[0]
        return u.cpu().numpy()


def box_points(scan, box):
    R = scan["rows"]
    m = ((R[:, 0] >= box["c"][0] - 1e-9) & (R[:, 1] >= box["phi"][0] - 1e-9)
         & (R[:, 2] <= box["kh"][1] + 1e-9) & R[:, 3].astype(bool))
    return np.where(m)[0]


def evaluate(model, scan, gdef, device, full=True, eb=None):
    box = gdef["box"]
    ii = box_points(scan, box)
    R = scan["rows"][ii]
    P = R[:, :3]
    uc_f = R[:, 4:6]
    uc_p = crest_pred(model, P, device, eb)
    E_f, E_p = E_of(uc_f, P[:, 2], gdef), E_of(uc_p, P[:, 2], gdef)

    rng = np.random.default_rng(0)
    cal = rng.choice(len(ii), N_CAL, replace=False)
    test = np.setdiff1d(np.arange(len(ii)), cal)
    b = (uc_f[cal] - uc_p[cal]).mean(0)
    E_c = E_of(uc_p + b, P[:, 2], gdef)
    u_el = np.array(gdef["u_el0_m"]) + P[:, 2:3] * np.array(gdef["du_el_dkh_m"])
    rf, rp = uc_f - u_el, uc_p - u_el
    s_amp = float((rf[cal] * rp[cal]).sum() / max((rp[cal] ** 2).sum(), 1e-30))
    E_s = np.linalg.norm(s_amp * rp, axis=1) * 1e3 / gdef["U_ALARM_mm"] - 1.0
    clear = test[np.abs(E_f[test]) >= NOISE_E]
    out = dict(n=len(ii), n_test_clear=len(clear),
               sign_raw=float(np.mean(np.sign(E_p[clear]) == np.sign(E_f[clear]))),
               sign_cal=float(np.mean(np.sign(E_c[clear]) == np.sign(E_f[clear]))),
               offset_mm=(b * 1e3).tolist(),
               sign_scale=float(np.mean(np.sign(E_s[clear]) == np.sign(E_f[clear]))),
               scale=s_amp, E_scale=E_s.tolist(),
               crest_err_mm_median=float(np.median(np.linalg.norm(uc_p - uc_f, axis=1)) * 1e3),
               E_fem=E_f.tolist(), E_pinn=E_p.tolist(), E_cal=E_c.tolist(), params=P.tolist())
    if not full:
        return out

    X = torch.tensor(scan["coords"], dtype=DTYPE, device=device)
    Cn = torch.tensor(scan["centroids"], dtype=DTYPE, device=device)
    bX = Base(eb, scan["coords"], device, grad=False) if eb is not None else None
    bC = Base(eb, scan["centroids"], device) if eb is not None else None
    M1, M3, plf = [], [], []
    for j, i in enumerate(ii):
        pr = torch.tensor(P[j], dtype=DTYPE, device=device)
        with torch.no_grad():
            U = model(X, pr.expand(len(X), 3))
            if bX is not None:
                U = U + bX.at(pr.expand(len(X), 3)[:, 2])[0]
            U = U.cpu().numpy()
        Uf = scan["U"][i]
        M1.append(np.linalg.norm(U - Uf) / np.linalg.norm(Uf))
        prc = pr.expand(len(Cn), 3)
        eps, _ = strain(model, Cn, prc, create_graph=False,
                        base=None if bC is None else bC.at(prc[:, 2]))
        _, _, dg, _ = en.stress(eps.detach(), material(prc))
        A, B = scan["dgam"][i] > DGAM_THR, dg.cpu().numpy() > DGAM_THR
        M3.append((A & B).sum() / max((A | B).sum(), 1))
        plf.append(A.mean())
    M1, M3, plf = map(np.array, (M1, M3, plf))
    out.update(M1=M1.tolist(), M3=M3.tolist(), plastic_frac_fem=plf.tolist(),
               G1=float(np.median(M1)), G2=float(np.median(M3[plf >= 0.01])) if (plf >= 0.01).any() else float("nan"),
               n_G2=int((plf >= 0.01).sum()))
    return out


def report(r):
    print(f"\n  48 điểm hộp; điểm kiểm tra ngoài dải nhiễu: {r['n_test_clear']}")
    print(f"  sai số u_đỉnh PINN (trung vị) = {r['crest_err_mm_median']:.4f} mm"
          f"   (U_ALARM = {r.get('U_ALARM', float('nan'))} mm)")
    print(f"  độ lệch hiệu chỉnh b = ({r['offset_mm'][0]:+.4f}, {r['offset_mm'][1]:+.4f}) mm")
    print(f"  hệ số hiệu chỉnh s = {r['scale']:.3f}")
    print(f"  đúng dấu E: trước hiệu chỉnh {r['sign_raw']:.1%},  theo hệ số {r['sign_scale']:.1%},"
          f"  theo độ lệch hằng {r['sign_cal']:.1%}")
    if "G1" in r:
        g = [("G1 trung vị M1", r["G1"], "<=", 0.05),
             (f"G2 trung vị M3 ({r['n_G2']} điểm)", r["G2"], ">=", 0.60),
             ("G3 đúng dấu (không hiệu chỉnh)", r["sign_raw"], ">=", 0.90)]
        ok = True
        for name, v, op, t in g:
            good = v <= t if op == "<=" else v >= t
            ok &= bool(good)
            print(f"  {name:32s} = {v:.4f}  {op} {t}  {'ĐẠT' if good else 'KHÔNG ĐẠT'}")
        print("  => cổng 2 " + ("ĐẠT" if ok else "KHÔNG ĐẠT"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--nsub", type=int, default=2400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--residual", action="store_true", help="mạng chỉ học phần vượt đàn hồi")
    ap.add_argument("--u0", type=float, default=None, help="thang đầu ra của mạng (m); mặc định 0.01, hoặc 0.001 nếu --residual")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--monitor-every", type=int, default=2000,
                    help="in đúng dấu E ở đỉnh mỗi N bước (chỉ theo dõi, không dùng để chọn)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    scan = {k: v for k, v in np.load(os.path.join(RES, "scan_gate2.npz")).items()}
    gdef = json.load(open(os.path.join(RES, "gate2_def.json")))
    print(f"PINN tham số  hộp={gdef['box']}  U_ALARM={gdef['U_ALARM_mm']} mm  "
          f"seed={a.seed}  thiết bị={a.device}")

    eb = None
    if a.residual:
        from slope.elastic_field import ElasticBase
        eb = ElasticBase()
    u0 = a.u0 if a.u0 is not None else (0.001 if a.residual else 0.01)
    print(f"  chế độ: {'phần dư (u = u_el + u_res)' if a.residual else 'trường tổng'},  U0 = {u0*1e3:g} mm")

    def monitor(model, it):
        r = evaluate(model, scan, gdef, a.device, full=False, eb=eb)
        print(f"      [theo dõi {it}] sai số u_đỉnh {r['crest_err_mm_median']:.4f} mm"
              f"  đúng dấu trước/hệ số/độ lệch {r['sign_raw']:.0%}/{r['sign_scale']:.0%}/{r['sign_cal']:.0%}"
              f"  s={r['scale']:.2f}")

    model, secs = train(gdef["box"], a.steps, a.P, a.nsub, a.lr, a.width, a.depth, a.h,
                        a.seed, a.device, monitor=monitor, monitor_every=a.monitor_every,
                        eb=eb, u0=u0)
    r = evaluate(model, scan, gdef, a.device, full=True, eb=eb)
    r.update(U_ALARM=gdef["U_ALARM_mm"], seconds=secs, args=vars(a))
    report(r)
    name = a.name or f"pinn_param_seed{a.seed}"
    torch.save(model.state_dict(), os.path.join(RES, f"{name}.pt"))
    json.dump(r, open(os.path.join(RES, f"{name}.json"), "w"), indent=1)
    print(f"  đã lưu {os.path.join(RES, name)}.pt / .json  ({secs:.0f} s huấn luyện)")


if __name__ == "__main__":
    main()