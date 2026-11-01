"""
slope/scan.py — quét FEM trên lưới hộp tham số (c, phi, kh) của cổng 2.

Chạy trong fenicsx-env, giới hạn luồng để không làm chậm lần huấn luyện GPU
đang chạy song song:
    OMP_NUM_THREADS=1 $FEMPY -m slope.scan              # 5x5x5 = 125 điểm
    $FEMPY -m slope.scan --analyse-only                   # đọc lại kết quả đã có

Ba việc:
  1. Tìm chỗ KHÔNG có cân bằng trong hộp. Ở đó năng lượng dẻo lý tưởng không bị
     chặn dưới, PINN năng lượng sẽ phân kỳ nếu hộp huấn luyện chứa chỗ đó.
  2. Lấy số gia chuyển vị đỉnh so với trạng thái lắp cảm biến p0, để chọn U_ALARM.
  3. Lưu trường u và dgam của mọi điểm: tập kiểm tra đầu tiên cho cổng 2.

Quy tắc chốt trước:
  - p0 = góc bền nhất của hộp (c max, phi max, kh = 0). Không hội tụ ở p0 thì dừng.
  - Hộp huấn luyện cổng 2 = hộp con lớn nhất mà MỌI điểm lưới trong đó hội tụ.
  - U_ALARM = trung vị của |du_đỉnh| trên các điểm hội tụ, làm tròn 2 chữ số có
    nghĩa, để mặt E = 0 cắt qua giữa hộp. Đây là lựa chọn thiết kế cho benchmark,
    không phải ngưỡng kỹ thuật; phải ghi rõ như vậy trong luận văn.
"""
import argparse, itertools, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AXES = dict(c=np.linspace(6.0, 14.0, 5),
            phi=np.linspace(16.0, 24.0, 5),
            kh=np.linspace(0.0, 0.10, 5))
OUT = "scan_gate2.npz"


def run(solve, out_dir):
    grid = list(itertools.product(AXES["c"], AXES["phi"], AXES["kh"]))
    rows, U, DG, t0 = [], [], [], time.time()
    coords = centroids = None
    for i, (c, phi, kh) in enumerate(grid):
        o = solve(c=c, phi=phi, kh=kh, verbose=False)
        ok = bool(o["converged"])
        rows.append([c, phi, kh, ok, *np.asarray(o["u_crest"])[:2],
                     float(np.mean(o["dgam"] > 5e-4)), o["seconds"]])
        U.append(o["u"] if ok else np.full_like(o["u"], np.nan))
        DG.append(o["dgam"] if ok else np.full_like(o["dgam"], np.nan))
        coords, centroids = o["coords"], o["centroids"]
        print(f"  {i+1:3d}/{len(grid)} c={c:5.2f} phi={phi:5.2f} kh={kh:.3f} "
              f"{'hội tụ' if ok else 'KHÔNG cân bằng'}  "
              f"|u_đỉnh|={np.linalg.norm(rows[-1][4:6])*1e3:7.3f} mm  "
              f"dẻo={rows[-1][6]:5.1%}  {o['seconds']:.1f}s  ({time.time()-t0:.0f}s)")
    path = os.path.join(out_dir, OUT)
    np.savez_compressed(path, rows=np.array(rows, float), U=np.array(U),
                        dgam=np.array(DG), coords=coords, centroids=centroids,
                        axes=json.dumps({k: v.tolist() for k, v in AXES.items()}))
    print(f"  đã lưu {path}")
    return path


def largest_feasible_box(ok):
    """ok: mảng bool (nc, nphi, nkh). Hộp con liên tục lớn nhất (theo số điểm) toàn True."""
    best, n = None, ok.shape
    for a0, a1 in itertools.combinations_with_replacement(range(n[0]), 2):
        for b0, b1 in itertools.combinations_with_replacement(range(n[1]), 2):
            for k0, k1 in itertools.combinations_with_replacement(range(n[2]), 2):
                if ok[a0:a1+1, b0:b1+1, k0:k1+1].all():
                    size = (a1-a0+1) * (b1-b0+1) * (k1-k0+1)
                    if best is None or size > best[0]:
                        best = (size, (a0, a1), (b0, b1), (k0, k1))
    return best


def analyse(path):
    d = np.load(path)
    R = d["rows"]
    ax = {k: np.array(v) for k, v in json.loads(str(d["axes"])).items()}
    n = tuple(len(ax[k]) for k in ("c", "phi", "kh"))
    ok = R[:, 3].astype(bool).reshape(n)
    print(f"\n  hội tụ: {ok.sum()}/{ok.size} điểm")
    if not ok.all():
        bad = R[~R[:, 3].astype(bool)][:, :3]
        print(f"  không có cân bằng tại {len(bad)} điểm (c, phi, kh), in tối đa 15:")
        for b in bad[:15]:
            print(f"    {b[0]:5.2f} {b[1]:5.2f} {b[2]:.3f}")

    i0 = np.where((R[:, 0] == ax["c"].max()) & (R[:, 1] == ax["phi"].max())
                  & (R[:, 2] == ax["kh"].min()))[0][0]
    if not R[i0, 3]:
        print("  p0 KHÔNG hội tụ: dừng, xem lại hộp tham số")
        return
    u0 = R[i0, 4:6]
    du = np.linalg.norm(R[:, 4:6] - u0, axis=1) * 1e3          # mm
    conv = R[:, 3].astype(bool)
    print(f"  p0 = (c={R[i0,0]:.0f}, phi={R[i0,1]:.0f}, kh={R[i0,2]:.2f}),"
          f"  u_đỉnh(p0) = {np.linalg.norm(u0)*1e3:.3f} mm")
    q = np.percentile(du[conv], [0, 10, 50, 90, 100])
    print(f"  |du_đỉnh| (mm) trên điểm hội tụ: min {q[0]:.3f}  p10 {q[1]:.3f}"
          f"  trung vị {q[2]:.3f}  p90 {q[3]:.3f}  max {q[4]:.3f}")
    ua = float(f"{q[2]:.2g}")
    E = du / ua - 1.0
    print(f"  => U_ALARM đề xuất = {ua} mm;  tỉ lệ điểm E > 0: {(E[conv] > 0).mean():.1%}")

    best = largest_feasible_box(ok)
    (a0, a1), (b0, b1), (k0, k1) = best[1:]
    print(f"  hộp huấn luyện cổng 2 (mọi điểm lưới hội tụ, {best[0]}/{ok.size} điểm):")
    print(f"    c   {ax['c'][a0]:.1f} .. {ax['c'][a1]:.1f}")
    print(f"    phi {ax['phi'][b0]:.1f} .. {ax['phi'][b1]:.1f}")
    print(f"    kh  {ax['kh'][k0]:.3f} .. {ax['kh'][k1]:.3f}")


def self_test():
    """Kiểm phần phân tích bằng một hàm giả, không cần dolfinx."""
    import tempfile

    def fake(c, phi, kh, verbose=False):
        fos = (c / 10) * (np.tan(np.radians(phi)) / np.tan(np.radians(20))) * 1.47 / (1 + 8 * kh)
        ok = fos > 1.0
        u = np.array([1e-3 * (1 + 5 * kh), -7e-3 - 1e-3 / max(fos - 0.9, 0.05)])
        return dict(converged=ok, u_crest=u, u=np.zeros((4, 2)), dgam=np.zeros(3),
                    coords=np.zeros((4, 2)), centroids=np.zeros((3, 2)), seconds=0.0)
    with tempfile.TemporaryDirectory() as t:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            p = run(fake, t)
        analyse(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    from slope import oracle
    path = os.path.join(oracle.OUT_DIR, OUT)
    if not a.analyse_only:
        os.makedirs(oracle.OUT_DIR, exist_ok=True)
        path = run(oracle.solve, oracle.OUT_DIR)
    analyse(path)


if __name__ == "__main__":
    main()