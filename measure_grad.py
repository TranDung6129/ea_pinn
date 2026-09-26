"""
measure_grad.py — đo đóng góp realtime của mô hình: ∂E/∂p bằng autodiff.

  python measure_grad.py --tag cap --variant ls1_bs0_kn0
  python measure_grad.py --tag cap --variant ls1_bs0_kn0 --n-points 50 --h 0.01,0.02,0.05

Ba câu hỏi tách riêng, đừng trộn:
  1. autodiff có tự nhất quán không     → so với sai phân hữu hạn trên chính PINN
                                          (phải sát máy; nếu lệch là code sai, không phải mô hình sai)
  2. gradient có đúng với hệ thật không → so với sai phân hữu hạn trên oracle FEM
                                          (đây là con số dùng được, gồm cả sai số của PINN)
  3. nhanh hơn oracle bao nhiêu lần     → độ trễ của E, và của E kèm ∇E

Điểm đo lấy trên biên thật: nội suy tuyến tính theo α tại mỗi (β, D) nơi E_fem đổi dấu.
Điểm nào lệch một bước h là ra ngoài miền thì bỏ, để sai phân trung tâm còn đối xứng.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import config as cfg
from src.ground_truth import generate_ground_truth
from src.pinn_model import build_model, normalise_params
from src.loss_functions import failure_functional
from src.fem_oracle import batch_oracle

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="cap")
ap.add_argument("--seed", type=int, default=cfg.BASE_SEED)
ap.add_argument("--variant", default="ls1_bs0_kn0")
ap.add_argument("--n-points", type=int, default=50)
ap.add_argument("--h", default="0.01,0.02,0.05",
                help="bước sai phân, tính theo tỉ lệ của bề rộng từng trục")
ap.add_argument("--sigma", type=float, default=0.05,
                help="sai số cảm biến tương đối, dùng cho phần truyền bất định")
ap.add_argument("--jobs", type=int, default=None)
args = ap.parse_args()
dev = cfg.DEVICE
H_REL = [float(x) for x in args.h.split(",")]


# ── nạp mô hình ──────────────────────────────────────────────────────────────
def load_model():
    name = f"fmd_pinn_seed{args.seed}_{args.variant}" + (f"_{args.tag}" if args.tag else "")
    path = os.path.join(cfg.CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        cands = [f for f in os.listdir(cfg.CKPT_DIR)
                 if f.startswith(f"{name}_call") and f.endswith(".pt")]
        if not cands:
            sys.exit(f"không thấy checkpoint cho {name}")
        path = os.path.join(cfg.CKPT_DIR,
                            max(cands, key=lambda f: int(f.rsplit("call", 1)[1][:-3])))
    m = build_model(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    print(f"  checkpoint: {os.path.basename(path)}")
    return m


# ── điểm đo trên biên thật ───────────────────────────────────────────────────
def boundary_points(gt, n, h_abs, rng):
    al, be, Dg, E = gt["alpha_grid"], gt["beta_grid"], gt["D_grid"], gt["E_fem"]
    lo = np.array([al.min(), be.min(), Dg.min()]) + h_abs
    hi = np.array([al.max(), be.max(), Dg.max()]) - h_abs
    pts = []
    for j in range(len(be)):
        for k in range(len(Dg)):
            col = E[:, j, k]
            s = np.where(np.sign(col[:-1]) * np.sign(col[1:]) < 0)[0]
            for i in s:
                t = col[i] / (col[i] - col[i + 1])
                p = np.array([al[i] + t * (al[i + 1] - al[i]), be[j], Dg[k]])
                if np.all(p > lo) and np.all(p < hi):
                    pts.append(p)
    pts = np.array(pts)
    if len(pts) == 0:
        sys.exit("không tìm được điểm biên nào còn đủ lề cho bước h lớn nhất")
    idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    return pts[idx]


# ── E và ∇E ──────────────────────────────────────────────────────────────────
def E_only(model, pts):
    with torch.no_grad():
        p = normalise_params(*[torch.tensor(pts[:, i], dtype=torch.float32, device=dev)
                               for i in range(3)])
        return failure_functional(model, p, device=dev).cpu().numpy()


def E_and_grad(model, pts):
    """∇E theo đơn vị vật lý: chuỗi đạo hàm đi qua normalise_params là tự động."""
    t = [torch.tensor(pts[:, i], dtype=torch.float32, device=dev, requires_grad=True)
         for i in range(3)]
    E = failure_functional(model, normalise_params(*t), device=dev)
    g = torch.autograd.grad(E.sum(), t)          # E_i chỉ phụ thuộc p_i nên sum() là đủ
    return E.detach().cpu().numpy(), torch.stack(g, 1).cpu().numpy()


def fd_pinn(model, pts, h_abs):
    g = np.zeros_like(pts)
    for i in range(3):
        for s in (1.0, -1.0):
            q = pts.copy(); q[:, i] += s * h_abs[i]
            g[:, i] += s * E_only(model, q) / (2 * h_abs[i])
    return g


def fd_oracle(pts, h_abs):
    qs = []
    for i in range(3):
        for s in (1.0, -1.0):
            q = pts.copy(); q[:, i] += s * h_abs[i]; qs.append(q)
    E = np.asarray(batch_oracle(np.vstack(qs), show_progress=True, n_jobs=args.jobs),
                   dtype=float).reshape(6, len(pts))
    return np.stack([(E[2 * i] - E[2 * i + 1]) / (2 * h_abs[i]) for i in range(3)], 1)


SPAN = None   # gán trong __main__; dùng để đổi ∇E sang toạ độ chuẩn hoá


def compare(g_ref, g_test, label):
    """So trong toạ độ chuẩn hoá: ∂E/∂p̂ = ∂E/∂p · span.
    Theo đơn vị vật lý, ∂E/∂D lớn gấp ~20 lần ∂E/∂α chỉ vì miền của D hẹp hơn,
    nên cos theo đơn vị vật lý bị thành phần D chi phối và không đọc được."""
    a, b = g_ref * SPAN, g_test * SPAN
    nr, nt = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = nr > 1e-12
    cos = (a[ok] * b[ok]).sum(1) / (nr[ok] * nt[ok] + 1e-30)
    rel = np.abs(nt[ok] - nr[ok]) / nr[ok]
    print(f"    {label:<14s} cos: trung vị {np.median(cos):.4f}  p10 {np.percentile(cos,10):.4f}"
          f"   |∇E| tương đối: trung vị {np.median(rel):.3f}  p90 {np.percentile(rel,90):.3f}")
    for i, nm in enumerate(("α", "β", "D")):
        r = np.abs(b[ok, i] - a[ok, i]) / (np.abs(a[ok, i]) + 1e-12)
        s = np.mean(np.sign(a[ok, i]) == np.sign(b[ok, i]))
        print(f"        d/d{nm}: sai số tương đối trung vị {np.median(r):.3f}"
              f"   đúng dấu {s:.0%}   tham chiếu trung vị {np.median(a[ok, i]):+.4f}")
    return np.median(cos), np.median(rel)


# ── độ trễ ───────────────────────────────────────────────────────────────────
def latency(model, n, with_grad, reps=30):
    pts = np.column_stack([
        np.random.uniform(*cfg.ALPHA_RANGE, n),
        np.random.uniform(*cfg.BETA_RANGE, n),
        np.random.uniform(0.005, 0.3, n)])
    def f(chunk=256):
        for i in range(0, n, chunk):
            q = pts[i:i + chunk]
            E_and_grad(model, q) if with_grad else E_only(model, q)
    for _ in range(5):
        f()
    if dev == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ── chạy ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(args.seed)
    model = load_model()
    gt = generate_ground_truth()

    span = np.array([cfg.ALPHA_RANGE[1] - cfg.ALPHA_RANGE[0],
                     cfg.BETA_RANGE[1] - cfg.BETA_RANGE[0],
                     gt["D_grid"].max() - gt["D_grid"].min()])
    h_max = max(H_REL) * span
    SPAN = span
    pts = boundary_points(gt, args.n_points, h_max, rng)
    print(f"  {len(pts)} điểm trên biên thật  (β {pts[:,1].min():.2f}–{pts[:,1].max():.2f}, "
          f"D {pts[:,2].min():.4f}–{pts[:,2].max():.4f})")

    E_ad, g_ad = E_and_grad(model, pts)
    print(f"\n  |E_pinn| tại các điểm này: trung vị {np.median(np.abs(E_ad)):.4f}  "
          f"(lý tưởng là 0, vì đây là điểm trên biên thật)")

    print("\n1. autodiff so với sai phân trên chính PINN  (kiểm tra code, phải sát 1.0000)")
    for hr in H_REL:
        compare(fd_pinn(model, pts, hr * span), g_ad, f"h={hr:.3f}")

    print("\n2. autodiff so với sai phân trên oracle FEM  (con số dùng được)")
    res = {}
    for hr in H_REL:
        h_abs = hr * span
        g_fem = fd_oracle(pts, h_abs)
        res[hr] = (g_fem, *compare(g_fem, g_ad, f"h={hr:.3f}"))

    hr_best = min(res, key=lambda k: res[k][2])
    g_fem = res[hr_best][0]
    print(f"\n   truyền bất định cảm biến σ = {args.sigma:.0%} tương đối, mỗi trục độc lập")
    sig = args.sigma * np.abs(pts)
    sE_ad = np.sqrt(((g_ad * sig) ** 2).sum(1))
    sE_fem = np.sqrt(((g_fem * sig) ** 2).sum(1))
    rel = np.abs(sE_ad - sE_fem) / (sE_fem + 1e-30)
    print(f"    σ_E từ autodiff  trung vị {np.median(sE_ad):.4f}")
    print(f"    σ_E từ FEM       trung vị {np.median(sE_fem):.4f}   (h={hr_best:.3f})")
    print(f"    sai số tương đối trung vị {np.median(rel):.3f}  p90 {np.percentile(rel,90):.3f}")

    print("\n3. độ trễ")
    for n in (1, 32, 1024):
        t_e, t_g = latency(model, n, False), latency(model, n, True)
        print(f"    n={n:<5d}  E: {t_e*1e3:8.3f} ms ({t_e/n*1e6:7.1f} µs/điểm)"
              f"   E+∇E: {t_g*1e3:8.3f} ms ({t_g/n*1e6:7.1f} µs/điểm)")
    if dev == "cuda":
        print(f"    VRAM đỉnh: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB"
              f"  (trọng số {sum(p.numel() for p in model.parameters())*4/2**20:.2f} MiB)")
    t1 = latency(model, 1, True)
    print(f"\n    một lời gọi oracle FEM ≈ 20.9 ms (đã đo)")
    print(f"    E+∇E cho một cấu hình  = {t1*1e3:.3f} ms  →  nhanh hơn {20.9e-3/t1:.0f} lần")
    print(f"    và oracle không cho ∇E; muốn có phải thêm 6 lời gọi nữa ≈ 125 ms")