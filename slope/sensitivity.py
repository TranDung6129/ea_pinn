"""
slope/sensitivity.py — độ nhạy của đầu ra theo (c, phi, kh) và không gian con hoạt động.

Chạy trong fenicsx-env (gọi oracle trực tiếp):
    $FEMPY -m slope.sensitivity                 # 8 điểm, ~56 lời gọi, vài phút
    $FEMPY -m slope.sensitivity --n 12 --seed 1

Câu hỏi: tham số nào đáng làm đầu vào của PINN ở cổng 2?

Cách đo: tại N điểm ngẫu nhiên trong hộp tham số (toạ độ chuẩn hoá [-1, 1]),
tính gradient của q = |u_đỉnh| bằng sai phân trung tâm trên oracle. Từ đó:

  (1) Độ nhạy trung bình |dq/dp_i|: tham số nào gần như trơ (như D ở bài 2D).
  (2) Ma trận C = trung bình (grad q)(grad q)^T và phân tích trị riêng
      (không gian con hoạt động, Constantine 2015). Nếu một trị riêng áp đảo
      thì q chỉ phụ thuộc MỘT tổ hợp của các tham số: có thể gộp chúng làm một
      đầu vào (SRF chính là một tổ hợp như thế của c và tan phi). Vectơ riêng
      cho biết tổ hợp đó.
  (3) cos giữa các cột gradient qua các điểm: hai tham số đẩy q theo cùng một
      kiểu thì khó tách chúng ra từ q, tức khó nhận dạng từ cảm biến.

Ngưỡng quyết định, chốt trước khi chạy:
  - tham số có độ nhạy trung bình < 5% của tham số nhạy nhất: TRƠ, bỏ khỏi cổng 2
  - lambda_2 / lambda_1 < 0.01: q thực chất một chiều, cân nhắc gộp
  - còn lại: giữ riêng từng tham số
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NAMES = ("c", "phi", "kh")
BOX = dict(c=(6.0, 14.0), phi=(16.0, 24.0), kh=(0.0, 0.10))   # hộp dự kiến cho cổng 2
STEP = 0.05          # bước sai phân, theo toạ độ chuẩn hoá (5% nửa khoảng)


def to_phys(z):
    lo = np.array([BOX[n][0] for n in NAMES])
    hi = np.array([BOX[n][1] for n in NAMES])
    return lo + (np.asarray(z) + 1.0) * 0.5 * (hi - lo)


def analyse(G, names=NAMES):
    """G: (N, d) gradient theo toạ độ chuẩn hoá. Trả về dict các đại lượng quyết định."""
    G = np.asarray(G, float)
    mean_abs = np.abs(G).mean(0)
    share = mean_abs / mean_abs.max()
    C = G.T @ G / len(G)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    norms = np.linalg.norm(G, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (G.T @ G) / np.outer(norms, norms)
    return dict(mean_abs=mean_abs, share=share, eig=lam, vec=V, cos=cos,
                inert=[n for n, s in zip(names, share) if s < 0.05],
                one_dim=bool(lam[1] / lam[0] < 0.01) if len(lam) > 1 else True)


def report(r, names=NAMES):
    print("\n  Độ nhạy trung bình |dq/dp| (q = |u_đỉnh| mm, p chuẩn hoá [-1,1]):")
    for n, m, s in zip(names, r["mean_abs"], r["share"]):
        print(f"    {n:4s} {m:9.4f}   {s:6.1%} của lớn nhất   {'TRƠ' if s < 0.05 else ''}")
    print("  Trị riêng của C (không gian con hoạt động):")
    for i, (l, v) in enumerate(zip(r["eig"], r["vec"].T)):
        comb = " + ".join(f"{x:+.2f}·{n}" for x, n in zip(v, names))
        print(f"    lambda_{i+1} = {l:.4e}  ({l/r['eig'][0]:.2e} lần lambda_1)   hướng: {comb}")
    print("  cos giữa các cột gradient:")
    for i in range(len(names)):
        print("    " + "  ".join(f"{r['cos'][i, j]:+.2f}" for j in range(len(names))) + f"   {names[i]}")
    print("\n  Quyết định theo ngưỡng chốt trước:")
    print(f"    tham số trơ: {r['inert'] or 'không có'}")
    print(f"    q thực chất một chiều: {'CÓ, cân nhắc gộp' if r['one_dim'] else 'không, giữ riêng'}")


def self_test():
    """Kiểm phần phân tích trên hàm biết trước đáp án, không cần oracle."""
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1, 1, (50, 3))
    # q = (z0 + 2 z1)^2 : phụ thuộc đúng một tổ hợp, z2 trơ
    G = np.stack([2 * (Z[:, 0] + 2 * Z[:, 1]), 4 * (Z[:, 0] + 2 * Z[:, 1]), 0 * Z[:, 2]], 1)
    r = analyse(G)
    v = r["vec"][:, 0] * np.sign(r["vec"][0, 0])
    ok = (r["inert"] == ["kh"] and r["one_dim"]
          and np.allclose(v, np.array([1, 2, 0]) / np.sqrt(5), atol=1e-6))
    print(f"  tự kiểm phân tích: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}  (trơ={r['inert']}, "
          f"một chiều={r['one_dim']}, hướng={np.round(v, 3)})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if not self_test() or a.self_test:
        return

    from slope import oracle
    rng = np.random.default_rng(a.seed)
    # điểm gốc cổng 1 (c=10, phi=20, kh=0 -> z_kh = -1, đẩy vào trong để sai phân 2 phía)
    Z = np.vstack([[0.0, 0.0, -1.0 + 2 * STEP], rng.uniform(-0.8, 0.8, (a.n - 1, 3))])

    def q_of(z):
        c, phi, kh = to_phys(z)
        out = oracle.solve(c=c, phi=phi, kh=kh, verbose=False)
        if not out["converged"]:
            return np.nan, np.nan
        return out["u_crest_mag"] * 1e3, float(np.mean(out["dgam"] > 5e-4))

    rows, G, t0 = [], [], time.time()
    for i, z in enumerate(Z):
        q0, pl = q_of(z)
        g = np.full(3, np.nan)
        if np.isfinite(q0):
            for j in range(3):
                e = np.zeros(3); e[j] = STEP
                qp, _ = q_of(z + e)
                qm, _ = q_of(z - e)
                g[j] = (qp - qm) / (2 * STEP)
        c, phi, kh = to_phys(z)
        print(f"  điểm {i}: c={c:5.2f} phi={phi:5.2f} kh={kh:.3f}  |u_đỉnh|={q0:7.3f} mm"
              f"  dẻo={pl:5.1%}  grad={np.round(g, 4)}  ({time.time()-t0:.0f} s)")
        rows.append(dict(z=z.tolist(), p=[c, phi, kh], q=q0, plastic=pl, grad=g.tolist()))
        if np.all(np.isfinite(g)):
            G.append(g)

    print(f"\n  {len(G)}/{len(Z)} điểm dùng được (điểm không cân bằng bị bỏ)")
    if len(G) >= 3:
        r = analyse(np.array(G))
        report(r)
    path = os.path.join(oracle.OUT_DIR, f"sensitivity_seed{a.seed}.json")
    os.makedirs(oracle.OUT_DIR, exist_ok=True)
    json.dump(dict(box=BOX, step=STEP, rows=rows), open(path, "w"), indent=1)
    print(f"  đã lưu {path}")


if __name__ == "__main__":
    main()