"""
slope/gate2_def.py — chốt định nghĩa E cho cổng 2: chuyển vị DƯ (vượt đàn hồi) tại mép đỉnh.

    E(p) = |u_đỉnh(p) - u_đỉnh^el(kh)| / U_ALARM - 1,   p = (c, phi, kh)

u^el là lời giải ĐÀN HỒI THUẦN (cường độ vô hạn) cùng lưới, cùng tải. Vì đàn
hồi tuyến tính nên u^el(kh) = u^el(0) + kh/kh1 * (u^el(kh1) - u^el(0)): chỉ cần
hai lời giải FEM, lời giải thứ ba (kh1/2) để kiểm tính tuyến tính.
Phần trừ đi là biến dạng tự hồi phục; phần còn lại là chuyển vị vĩnh cửu
(tinh thần phân tích Newmark cho mái dốc chịu động đất).

Chạy trong fenicsx-env (3 lời giải đàn hồi, vài giây mỗi cái):
    python -m slope.oracle --c 1e6 --kh 0     --name elastic_kh0
    python -m slope.oracle --c 1e6 --kh 0.05  --name elastic_kh005
    python -m slope.oracle --c 1e6 --kh 0.1   --name elastic_kh01
rồi (môi trường nào cũng được, chỉ cần numpy):
    python -m slope.gate2_def

Hộp cổng 2 (chọn 27/09 sau khi xem quét, xem ghi chú trong hội thoại):
    c 8..14, phi 18..24, kh 0..0.05. Góc yếu nhất (8, 18, 0.05) dẻo 14.7%, không
    khó hơn trạng thái cổng 1 (16.7%). Trong ba hộp 48 điểm cùng đạt, hộp này đi
    sâu nhất vào vùng dẻo, nơi c và phi thật sự ảnh hưởng tới E.

Quy tắc ngưỡng, chốt TRƯỚC khi có lời giải đàn hồi:
    U_ALARM = trung vị |du_dư| trên 48 điểm của hộp, nhưng không nhỏ hơn
    10 x sàn rời rạc hoá FEM của u_đỉnh (0.0035 mm: lệch lưới 0.5 so với 0.25
    ở cổng 1 là 8.287 so với 8.284 mm), làm tròn 2 chữ số có nghĩa.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOX = dict(c=(8.0, 14.0), phi=(18.0, 24.0), kh=(0.0, 0.05))
FEM_FLOOR_MM = 0.0035
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_slope")


def load(name):
    d = np.load(os.path.join(RES, name) if not os.path.isabs(name) else name, allow_pickle=False)
    return {k: d[k] for k in d.files}


def elastic_reference(e0, e1, emid=None):
    """Trả (u0, du/dkh) của u^el tại mép đỉnh, đơn vị mét, và các kiểm tra."""
    for n, e in (("kh0", e0), ("kh1", e1)) + ((("mid", emid),) if emid else ()):
        if not bool(e["converged"]) or float(np.max(e["dgam"])) > 0.0:
            sys.exit(f"  lời giải đàn hồi {n} có chảy dẻo hoặc không hội tụ: tăng --c")
    kh1 = json.loads(str(e1["params"]))["kh"]
    u0, u1 = e0["u_crest"][:2], e1["u_crest"][:2]
    slope = (u1 - u0) / kh1
    if emid is not None:
        khm = json.loads(str(emid["params"]))["kh"]
        err = np.linalg.norm(emid["u_crest"][:2] - (u0 + khm * slope)) / np.linalg.norm(u1 - u0)
        print(f"  kiểm tuyến tính: lệch tương đối {err:.1e}  {'ĐẠT' if err < 1e-6 else 'KHÔNG ĐẠT'} (ngưỡng 1e-6)")
        if err >= 1e-6:
            sys.exit("  u^el không tuyến tính theo kh: xem lại")
    return u0, slope


def define(scan, u0, slope):
    R = scan["rows"]
    c, phi, kh, ok = R[:, 0], R[:, 1], R[:, 2], R[:, 3].astype(bool)
    inbox = ((c >= BOX["c"][0] - 1e-9) & (phi >= BOX["phi"][0] - 1e-9)
             & (kh <= BOX["kh"][1] + 1e-9))
    if not ok[inbox].all():
        sys.exit("  có điểm trong hộp không hội tụ")
    u_el = u0[None, :] + kh[:, None] * slope[None, :]
    du = np.linalg.norm(R[:, 4:6] - u_el, axis=1) * 1e3              # mm
    med = float(np.median(du[inbox]))
    ua = float(f"{max(med, 10 * FEM_FLOOR_MM):.2g}")
    E = du / ua - 1.0
    print(f"  hộp: {inbox.sum()} điểm.  |du_dư| (mm): min {du[inbox].min():.4f}"
          f"  trung vị {med:.4f}  max {du[inbox].max():.4f}")
    print(f"  U_ALARM = {ua} mm  ({'trung vị' if med >= 10 * FEM_FLOOR_MM else 'chạm sàn 10x FEM'})"
          f";  E > 0 ở {(E[inbox] > 0).mean():.0%} điểm hộp")

    # biên E = 0 có thật sự phụ thuộc c, phi không: vị trí kh* dọc từng cặp (c, phi)
    ks = []
    for cc in np.unique(c[inbox]):
        for pp in np.unique(phi[inbox]):
            m = inbox & (c == cc) & (phi == pp)
            o = np.argsort(kh[m]); k_, e_ = kh[m][o], E[m][o]
            s = np.where((e_[:-1] < 0) & (e_[1:] >= 0))[0]
            ks.append(k_[s[0]] + (0 - e_[s[0]]) / (e_[s[0]+1] - e_[s[0]]) * (k_[s[0]+1] - k_[s[0]])
                      if len(s) else (0.0 if e_[0] >= 0 else np.nan))
    ks = np.array(ks)
    print(f"  kh tại biên theo từng cặp (c, phi): có biên {np.isfinite(ks).sum()}/{ks.size},"
          f"  trải từ {np.nanmin(ks):.4f} đến {np.nanmax(ks):.4f} (dải kh của hộp 0..{BOX['kh'][1]})")
    out = dict(box=BOX, U_ALARM_mm=ua, u_el0_m=u0.tolist(), du_el_dkh_m=slope.tolist(),
               E_def="|u_crest(p) - (u_el0 + kh*du_el_dkh)| / U_ALARM - 1")
    path = os.path.join(RES, "gate2_def.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"  đã lưu {path}")
    return out


def main():
    scan = load("scan_gate2.npz")
    e0, e1 = load("elastic_kh0.npz"), load("elastic_kh01.npz")
    em = load("elastic_kh005.npz") if os.path.exists(os.path.join(RES, "elastic_kh005.npz")) else None
    u0, slope = elastic_reference(e0, e1, em)
    define(scan, u0, slope)


if __name__ == "__main__":
    main()