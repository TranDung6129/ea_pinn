"""
slope/oracle.py — oracle FEM cho mái dốc đàn dẻo, biến dạng phẳng, dolfinx 0.9.

Chạy trong môi trường fenicsx-env (KHÔNG cần torch):

    python -m slope.oracle                 # một lời gọi với bộ tham số mặc định
    python -m slope.oracle --plot          # kèm hình trường chuyển vị và vùng dẻo
    python -m slope.oracle --check-steps   # tự kiểm: kết quả không phụ thuộc số bước tải
    python -m slope.oracle --check-fos     # tự kiểm: hệ số an toàn so với Griffiths & Lane
    python -m slope.oracle --c 8 --phi 18  # đổi tham số

Kết quả ghi ra results_slope/<tên>.npz: toạ độ nút P2, chuyển vị, tâm phần tử
và độ lớn chảy dẻo dgam trên từng phần tử, u_đỉnh, E, cờ hội tụ, số vòng Newton.
Phía PINN chỉ đọc file này, không import dolfinx.

Mô hình vật liệu: Hencky + Drucker-Prager kết hợp, xem slope/constitutive.py.
Hàm _sigma dưới đây là bản UFL của đúng công thức đó; --check-constitutive so
hai bản trên các biến dạng lấy từ chính lời giải.

Ghi chú "chuyển vị bật trọng lực"
---------------------------------
Trọng lực được bật từ trạng thái không ứng suất, nên u ở đây gồm cả độ lún đàn
hồi do trọng lượng bản thân, một đại lượng phụ thuộc mạnh vào YOUNG và không
phải thứ cảm biến ngoài hiện trường đo được (cảm biến đo từ lúc lắp đặt). Với
cổng 1 điều này không sao: PINN và FEM giải cùng một bài, so cùng một u. Trước
khi chốt U_ALARM cho cổng 2, cần định nghĩa E theo SỐ GIA chuyển vị giữa một
trạng thái tham chiếu và trạng thái bị kích hoạt (ví dụ mực nước dâng).
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slope import config_slope as cs

from mpi4py import MPI
from petsc4py import PETSc
import gmsh
import ufl
from dolfinx import fem, geometry, mesh as dmesh
from dolfinx.io import gmshio
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results_slope")

# Nhãn biên vật lý trong gmsh
TAG_BOTTOM, TAG_LEFT, TAG_RIGHT, TAG_SURFACE = 1, 2, 3, 4


# ── Hình học và lưới ────────────────────────────────────────────────────────

def slope_points(H=None, run=None):
    """Đa giác miền, đi ngược chiều kim đồng hồ. Mặt dốc đi xuống về bên phải."""
    H = H or cs.H_SLOPE
    run = run or cs.SLOPE_RUN
    x_crest = cs.L_CREST
    x_toe = x_crest + run
    W = x_toe + cs.L_TOE
    y_toe, y_top = cs.H_FOUND, cs.H_FOUND + H
    pts = [(0.0, 0.0), (W, 0.0), (W, y_toe), (x_toe, y_toe),
           (x_crest, y_top), (0.0, y_top)]
    return pts, (x_crest, y_top), (x_toe, y_toe), W


def build_mesh(H=None, run=None, h=None):
    h = h or cs.MESH_H
    pts, crest, toe, W = slope_points(H, run)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("slope")
    p = [gmsh.model.geo.addPoint(x, y, 0.0, h) for x, y in pts]
    lines = [gmsh.model.geo.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    # cạnh: 0 đáy, 1 phải, 2..4 mặt nền + mặt dốc + đỉnh (mặt tự do), 5 trái
    gmsh.model.addPhysicalGroup(1, [lines[0]], TAG_BOTTOM)
    gmsh.model.addPhysicalGroup(1, [lines[1]], TAG_RIGHT)
    gmsh.model.addPhysicalGroup(1, [lines[5]], TAG_LEFT)
    gmsh.model.addPhysicalGroup(1, lines[2:5], TAG_SURFACE)
    gmsh.model.addPhysicalGroup(2, [surf], 1)
    # lưới mịn hơn quanh mép đỉnh và chân dốc, nơi vùng dẻo hình thành
    f = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f, "PointsList", [p[3], p[4]])
    t = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(t, "InField", f)
    gmsh.model.mesh.field.setNumber(t, "SizeMin", h / 2.5)
    gmsh.model.mesh.field.setNumber(t, "SizeMax", h)
    gmsh.model.mesh.field.setNumber(t, "DistMin", 1.0)
    gmsh.model.mesh.field.setNumber(t, "DistMax", 6.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(t)
    gmsh.model.mesh.generate(2)
    domain, _, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return domain, facet_tags, crest, toe, W


# ── Vật liệu: bản UFL của slope/constitutive.py ──────────────────────────────

def _eps3(u):
    e = ufl.sym(ufl.grad(u))
    return ufl.as_matrix([[e[0, 0], e[0, 1], 0.0],
                          [e[1, 0], e[1, 1], 0.0],
                          [0.0, 0.0, 0.0]])


def _sigma(u, mat):
    """Trả (sig 3x3, dgam, cờ đỉnh nón). Cùng công thức với constitutive.stress."""
    return _sigma_eps(_eps3(u), mat)


def _sigma_eps(eps, mat):
    """Như _sigma nhưng nhận thẳng tensor biến dạng 3x3 (để tự kiểm không cần lưới)."""
    lam, G, K, a, k = mat["lam"], mat["G"], mat["K"], mat["a"], mat["k"]
    I = ufl.Identity(3)
    sig_tr = lam * ufl.tr(eps) * I + 2.0 * G * eps
    I1 = ufl.tr(sig_tr)
    s = sig_tr - I1 / 3.0 * I
    # eps nhỏ trong căn chỉ để đạo hàm không vô hạn khi s = 0; đơn vị kPa^2
    sqJ2 = ufl.sqrt(0.5 * ufl.inner(s, s) + 1e-12)
    f = sqJ2 + a * I1 - k
    dgam = ufl.max_value(f, 0.0) / (G + 9.0 * K * a * a)
    q_new = sqJ2 - G * dgam
    cone = s * (q_new / sqJ2) + (I1 - 9.0 * K * a * dgam) / 3.0 * I
    apex_sig = (k / (3.0 * a)) * I
    is_apex = ufl.lt(q_new, 0.0)
    sig = ufl.conditional(is_apex, apex_sig, cone)
    return sig, dgam, is_apex


def material(c=None, phi=None, young=None, nu=None, srf=1.0):
    """Hằng số vật liệu. srf chia c và tan(phi) như phương pháp giảm cường độ."""
    c = (cs.COHESION if c is None else c) / srf
    phi = cs.PHI_DEG if phi is None else phi
    phi = np.degrees(np.arctan(np.tan(np.radians(phi)) / srf))
    young = cs.YOUNG if young is None else young
    nu = cs.POISSON if nu is None else nu
    G = young / (2 * (1 + nu))
    K = young / (3 * (1 - 2 * nu))
    t = np.tan(np.radians(phi))
    den = np.sqrt(9 + 12 * t * t)
    return dict(c=c, phi=phi, young=young, nu=nu, lam=K - 2 * G / 3, G=G, K=K,
                a=t / den, k=3 * c / den)


# ── Một lời gọi oracle ──────────────────────────────────────────────────────

def solve(c=None, phi=None, gamma=None, H=None, run=None, srf=1.0, kh=0.0,
          load_steps=None, mesh_h=None, verbose=True):
    """
    Giải cân bằng div(sig(u)) + b = 0 với b = (0, -lam_load*gamma), lam_load
    đi từ 0 tới 1 theo load_steps. Hencky: các bước chỉ là tiếp nối cho Newton.
    Nếu một bước không hội tụ thì chia đôi số gia; số gia nhỏ hơn MIN_DLOAD mà
    vẫn không hội tụ thì kết luận KHÔNG CÓ CÂN BẰNG ở mức tải đó.
    """
    t0 = time.time()
    gamma = cs.GAMMA if gamma is None else gamma
    domain, facet_tags, crest, toe, W = build_mesh(H, run, mesh_h)
    mat = material(c, phi, srf=srf)

    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)

    # Biên: đáy ngàm, hai bên trượt đứng (u_x = 0), mặt trên tự do
    fdim = domain.topology.dim - 1
    bottom = facet_tags.find(TAG_BOTTOM)
    sides = np.concatenate([facet_tags.find(TAG_LEFT), facet_tags.find(TAG_RIGHT)])
    bcs = [fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType),
                           fem.locate_dofs_topological(V, fdim, bottom), V),
           fem.dirichletbc(PETSc.ScalarType(0.0),
                           fem.locate_dofs_topological(V.sub(0), fdim, sides), V.sub(0))]

    lam_load = fem.Constant(domain, PETSc.ScalarType(0.0))
    # kh: hệ số động đất giả tĩnh, lực khối ngang hướng ra phía mặt dốc (+x),
    # tăng cùng trọng lực theo lam_load (tải tỉ lệ, đúng giả thiết Hencky)
    b = ufl.as_vector([kh * gamma * lam_load, -gamma * lam_load])
    sig, dgam, is_apex = _sigma(u, mat)
    dx = ufl.Measure("dx", domain=domain,
                     metadata={"quadrature_degree": cs.QUAD_DEGREE})
    # sig_zz không sinh công vì eps_zz = 0; UFL không cho cắt lát nên dựng khối 2x2 tay
    sig2 = ufl.as_matrix([[sig[0, 0], sig[0, 1]], [sig[1, 0], sig[1, 1]]])
    F = ufl.inner(sig2, ufl.sym(ufl.grad(v))) * dx - ufl.inner(b, v) * dx

    problem = NonlinearProblem(F, u, bcs=bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "residual"
    solver.rtol, solver.atol = cs.NEWTON_RTOL, cs.NEWTON_ATOL
    solver.max_it = cs.NEWTON_MAXIT
    solver.report = False
    solver.error_on_nonconvergence = False
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    pre = ksp.getOptionsPrefix()
    opts[f"{pre}ksp_type"] = "preonly"
    opts[f"{pre}pc_type"] = "lu"
    opts[f"{pre}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    steps = list(load_steps or cs.LOAD_STEPS)
    reached, total_newton, log = 0.0, 0, []
    u_prev = u.x.array.copy()
    target_i = 0
    while target_i < len(steps):
        target = steps[target_i]
        lam_load.value = target
        try:
            n_it, conv = solver.solve(u)
        except RuntimeError:
            n_it, conv = cs.NEWTON_MAXIT, False
        total_newton += n_it
        if conv:
            u.x.scatter_forward()
            u_prev = u.x.array.copy()
            log.append((float(target), int(n_it)))
            reached = target
            target_i += 1
            continue
        # không hội tụ: quay lại nghiệm trước, chèn bước giữa
        u.x.array[:] = u_prev
        u.x.scatter_forward()
        mid = 0.5 * (reached + target)
        if target - reached < cs.MIN_DLOAD:
            break
        steps.insert(target_i, mid)

    converged = reached >= 1.0 - 1e-12

    # u tại mép đỉnh (lấy điểm lệch vào trong một chút để chắc chắn nằm trong miền)
    u_crest = eval_point(domain, u, np.array([crest[0] + 1e-6, crest[1] - 1e-6]))

    # dgam trên từng phần tử (DG0), dùng để vẽ và so vùng dẻo
    Q = fem.functionspace(domain, ("DG", 0))
    dg = fem.Function(Q)
    dg.interpolate(fem.Expression(dgam, Q.element.interpolation_points()))
    ap = fem.Function(Q)
    ap.interpolate(fem.Expression(ufl.conditional(is_apex, 1.0, 0.0),
                                  Q.element.interpolation_points()))

    tdim = domain.topology.dim
    n_cells = domain.topology.index_map(tdim).size_local
    centroids = dmesh.compute_midpoints(domain, tdim, np.arange(n_cells, dtype=np.int32))[:, :2]
    cells_geom = domain.geometry.dofmap[:, :3]

    u_mag = float(np.linalg.norm(u_crest))
    out = dict(
        converged=converged, load_reached=reached, load_log=log,
        newton_total=total_newton, seconds=time.time() - t0,
        u_crest=u_crest, u_crest_mag=u_mag,
        E=(u_mag / cs.U_ALARM - 1.0) if converged else np.nan,
        coords=V.tabulate_dof_coordinates()[:, :2].copy(),
        u=u.x.array.reshape(-1, 2).copy(),
        centroids=centroids, dgam=dg.x.array[:n_cells].copy(),
        apex=ap.x.array[:n_cells].copy(),
        geom_x=domain.geometry.x[:, :2].copy(), cells=cells_geom.copy(),
        params=dict(c=mat["c"] * srf, phi=cs.PHI_DEG if phi is None else phi,
                    gamma=gamma, H=H or cs.H_SLOPE, run=run or cs.SLOPE_RUN,
                    srf=srf, kh=kh, young=mat["young"], nu=mat["nu"],
                    mesh_h=mesh_h or cs.MESH_H),
        crest=np.array(crest), toe=np.array(toe),
        n_dofs=V.dofmap.index_map.size_global * 2, n_cells=n_cells,
    )
    if verbose:
        print(f"  hội tụ={converged}  tải đạt={reached:.3f}  Newton={total_newton}"
              f"  |u_đỉnh|={u_mag*1e3:.2f} mm  E={out['E']:+.4f}"
              f"  phần tử dẻo={np.mean(out['dgam'] > 0):.1%}"
              f"  bậc tự do={out['n_dofs']}  {out['seconds']:.1f} s")
    return out


def eval_point(domain, u, xy):
    x = np.array([[xy[0], xy[1], 0.0]])
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, x)
    cells = geometry.compute_colliding_cells(domain, cand, x)
    if len(cells.links(0)) == 0:
        raise RuntimeError(f"điểm {xy} không nằm trong miền")
    val = u.eval(x, np.array([cells.links(0)[0]], dtype=np.int32))
    return np.asarray(val).reshape(-1)[:2].copy()     # 0.9 trả mảng phẳng


def save(out, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez_compressed(path, **{k: v for k, v in out.items()
                                 if k not in ("params", "load_log")},
                        params=json.dumps(out["params"]),
                        load_log=json.dumps(out["load_log"]))
    print(f"  đã lưu {path}")
    return path


# ── Tự kiểm ─────────────────────────────────────────────────────────────────

def check_steps():
    """Hencky không có lịch sử: 4 bước tải và 1 bước (nếu hội tụ) phải ra cùng u."""
    print("Tự kiểm 1: kết quả không phụ thuộc số bước tải")
    a = solve(load_steps=(0.25, 0.5, 0.75, 1.0))
    b = solve(load_steps=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    d = np.abs(a["u"] - b["u"]).max() / max(np.abs(a["u"]).max(), 1e-30)
    ok = d < 1e-6
    print(f"  lệch tương đối max |u| giữa 4 và 10 bước: {d:.2e}  "
          f"{'ĐẠT' if ok else 'KHÔNG ĐẠT'} (ngưỡng 1e-6)")
    return ok


def check_constitutive(out=None):
    """So sig của UFL với bản numpy tại các biến dạng lấy từ chính lời giải."""
    print("Tự kiểm 2: UFL khớp bản tham chiếu numpy")
    from slope import constitutive as cn
    rng = np.random.default_rng(0)
    e = cn.plane_strain(*rng.normal(0, 3e-4, (3, 2000)))
    # dựng một lưới 1 phần tử và nội suy biến dạng hằng để so điểm theo điểm
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    Q = fem.functionspace(msh, ("DG", 0, (3, 3)))
    mat = material()
    worst = 0.0
    for i in range(0, len(e), 50):
        exx, eyy, exy = e[i, 0, 0], e[i, 1, 1], e[i, 0, 1]
        u = fem.Function(V)
        u.interpolate(lambda x: np.vstack([exx * x[0] + exy * x[1],
                                           exy * x[0] + eyy * x[1]]))
        sig, _, _ = _sigma(u, mat)
        s = fem.Function(Q)
        s.interpolate(fem.Expression(sig, Q.element.interpolation_points()))
        s_ufl = s.x.array[:9].reshape(3, 3)
        s_np = cn.stress(e[i], mat["young"], mat["nu"], mat["c"], mat["phi"])
        worst = max(worst, np.abs(s_ufl - s_np).max() / mat["k"])
    ok = worst < 1e-6
    print(f"  max lệch / k = {worst:.2e}  {'ĐẠT' if ok else 'KHÔNG ĐẠT'} (ngưỡng 1e-6)")
    return ok


def fos(verbose=True):
    """Hệ số an toàn: SRF lớn nhất còn cân bằng, tìm bằng chia đôi."""
    lo, hi = cs.SRF_LO, cs.SRF_HI
    if not solve(srf=lo, verbose=False)["converged"]:
        raise RuntimeError(f"không cân bằng ngay cả ở SRF={lo}")
    while hi - lo > cs.SRF_TOL:
        mid = 0.5 * (lo + hi)
        ok = solve(srf=mid, verbose=False)["converged"]
        if verbose:
            print(f"    SRF={mid:.3f}  {'cân bằng' if ok else 'không cân bằng'}")
        lo, hi = (mid, hi) if ok else (lo, mid)
    return 0.5 * (lo + hi)


def check_fos():
    print("Tự kiểm 3: hệ số an toàn so với Griffiths & Lane (1999), ví dụ 1")
    F = fos()
    rel = abs(F - cs.FOS_REFERENCE) / cs.FOS_REFERENCE
    ok = rel < cs.FOS_TOLERANCE
    print(f"  FOS = {F:.3f}, tham chiếu {cs.FOS_REFERENCE}, lệch {rel:.1%}  "
          f"{'ĐẠT' if ok else 'KHÔNG ĐẠT'} (ngưỡng {cs.FOS_TOLERANCE:.0%})")
    return ok


# ── Vẽ ──────────────────────────────────────────────────────────────────────

def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(out["geom_x"][:, 0], out["geom_x"][:, 1], out["cells"])
    # chuyển vị trên nút đỉnh tam giác: lấy từ nút P2 trùng toạ độ
    key = {tuple(np.round(p, 9)): i for i, p in enumerate(out["coords"])}
    idx = np.array([key[tuple(np.round(p, 9))] for p in out["geom_x"]])
    umag = np.linalg.norm(out["u"][idx], axis=1) * 1e3
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    c0 = ax[0].tripcolor(tri, umag, shading="gouraud", cmap="viridis")
    fig.colorbar(c0, ax=ax[0], label="|u| (mm)")
    ax[0].set_title(f"Chuyển vị  |u_đỉnh| = {out['u_crest_mag']*1e3:.2f} mm")
    c1 = ax[1].tripcolor(tri, facecolors=out["dgam"] * 1e6, cmap="magma_r")
    fig.colorbar(c1, ax=ax[1], label="dgam × 1e6")
    ax[1].set_title(f"Vùng dẻo ({np.mean(out['dgam'] > 0):.1%} phần tử)")
    for a in ax:
        a.set_aspect("equal")
        a.plot(*out["crest"], "r^")
        a.plot(*out["toe"], "rv")
    fig.savefig(path, dpi=150)
    print(f"  đã lưu {path}")


# ── Dòng lệnh ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float)
    ap.add_argument("--phi", type=float)
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--srf", type=float, default=1.0)
    ap.add_argument("--kh", type=float, default=0.0)
    ap.add_argument("--mesh-h", type=float)
    ap.add_argument("--name", default="gate1_reference")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--check-steps", action="store_true")
    ap.add_argument("--check-constitutive", action="store_true")
    ap.add_argument("--check-fos", action="store_true")
    ap.add_argument("--check-all", action="store_true")
    args = ap.parse_args()

    if args.check_all or args.check_constitutive:
        check_constitutive()
    if args.check_all or args.check_steps:
        check_steps()
    if args.check_all or args.check_fos:
        check_fos()
    if args.check_all or args.check_steps or args.check_constitutive or args.check_fos:
        return

    out = solve(c=args.c, phi=args.phi, gamma=args.gamma, srf=args.srf, kh=args.kh,
                mesh_h=args.mesh_h)
    path = save(out, args.name)
    if args.plot:
        plot(out, path.replace(".npz", ".png"))


if __name__ == "__main__":
    main()
