"""
src/fmd_pinn.py
---------------
FMD-PINN: truy tìm mặt nghiệm dC = { p : E(p) = 0 }.

Phát biểu bài toán (đã sửa so với bản cũ):
  Bản cũ giải max_p E(p), tức đi tìm cấu hình SẬP NẶNG NHẤT. Điểm sập nặng
  nhất nằm sâu trong miền thất bại, không nằm trên biên, nên vòng đối kháng
  càng chạy càng rời xa thứ cần dựng. Đó là lý do mẫu bị dồn vô ích và FSR
  không xuống.

  Bản này giải E(p) = 0. Mọi lời gọi oracle đều đặt trên hoặc sát biên:
    1. Đề xuất: điểm trên tập mức không của PINN, xa các điểm đã gọi nhất.
    2. Xác nhận: bisection trên oracle giữa cặp bracket (E<0, E>0).
    3. Cấu hình tới hạn = điểm trên biên nơi ||grad_p E|| suy biến (định lý
       hàm ẩn: chỗ đó biên mất tính trơn). Vùng knee chính là loại điểm này,
       nên nó trở thành mục tiêu có định nghĩa chứ không phải chỗ mô hình
       tình cờ chạy kém.

Đã loại bỏ so với bản cũ:
  _outer_ascent (2 pha, multi-start LHS, lambda_div), _explore_safe_region,
  _build_anchor_replay, compute_sup_loss vòng for 19 mục, _emergency_retrain,
  _finetune, warmup 3 lần restart với ngưỡng chấp nhận, AdaptiveLossScheduler,
  AdaptiveSampler. Tất cả đều là miếng vá cho hậu quả của việc huấn luyện 500
  bước liên tiếp trên MỘT p rồi quên, và của mục tiêu max E.

Thay đổi cốt lõi về tốc độ:
  Huấn luyện theo BATCH THAM SỐ. ParametricPINN.forward đã nhận p_hat theo
  từng điểm từ đầu, nhưng code cũ chỉ đưa vào đúng một p. Nay mỗi bước phủ
  N_PARAM_BATCH tham số cùng lúc → không còn quên, không cần replay, và GPU
  được dùng đúng cách. Số bước giảm từ ~100k (200 call x 500 step) xuống
  ~12k, mỗi bước không còn np.argsort/KD-tree trên pool 81920 điểm.
"""

import torch
import torch.optim as optim
import numpy as np
import os, json, time
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config as cfg
from src.pinn_model import (build_model, normalise_params,
                            denormalise_params, clamp_normalised)
from src.loss_functions import (loss_pde, loss_bc, loss_ic,
                                loss_supervised, failure_functional)
from src.sampling import (lhs, lhs_region, clamp_region, region_bounds,
                          region_is_full, collocation, initial_points,
                          boundary_points)
from src.fem_oracle import solve_reaction_diffusion


# ── Siêu tham số (đọc từ config nếu có, không thì dùng mặc định) ─────────────
N_PARAM_BATCH  = getattr(cfg, "N_PARAM_BATCH", 32)    # số p mỗi bước
N_COLL_PER_P   = getattr(cfg, "N_COLL_PER_P", 256)    # điểm collocation mỗi p
N_SUP_BATCH    = getattr(cfg, "N_SUP_BATCH", 32)      # số bản ghi giám sát mỗi bước
LAMBDA_SUP     = getattr(cfg, "LAMBDA_SUP", 5.0)
INIT_STEPS     = getattr(cfg, "INIT_STEPS", 3000)     # huấn luyện ban đầu
STEPS_PER_CALL = getattr(cfg, "STEPS_PER_CALL", 60)   # sau mỗi oracle call
N_SEED_CALLS   = getattr(cfg, "N_SEED_CALLS", 10)     # LHS mồi để có bracket
SCAN_GRID      = getattr(cfg, "SCAN_GRID", 16)        # lưới quét tập mức không
BAND           = getattr(cfg, "SCAN_BAND", 0.15)
KNEE_WEIGHT    = getattr(cfg, "KNEE_WEIGHT", 0.5)
BISECT_EVERY   = getattr(cfg, "BISECT_EVERY", 3)      # cứ N call thì 1 lần bisection
CKPT_EVERY     = getattr(cfg, "CKPT_EVERY", 10)


class FMDPINNTrainer:
    """
    Cờ ablation (thay cho use_min_max / use_event_loss / use_adaptive_samp cũ):
      use_levelset  : đề xuất trên tập mức không của PINN.
                      False -> lấy mẫu LHS thụ động (baseline bị động).
      use_bisection : xen kẽ bisection trên oracle để sửa lệch của PINN.
      use_knee      : ưu tiên điểm biên có ||grad_p E|| nhỏ (cấu hình tới hạn).
    """

    def __init__(self, seed: int = 42,
                 use_levelset: bool = True,
                 use_bisection: bool = True,
                 use_knee: bool = True,
                 oracle_budget: int = None,
                 tag: str = ""):
        self.seed = seed
        self.use_levelset = use_levelset
        self.use_bisection = use_bisection
        self.use_knee = use_knee
        self.oracle_budget = oracle_budget or cfg.ORACLE_BUDGET
        self.tag = tag
        self.device = cfg.DEVICE

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = build_model(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.LR_ADAM)

        self.oracle_history: list = []
        self._resume_from = 0

        # Lưới quét tham số, cố định, dựng một lần
        self._scan_grid = self._build_scan_grid()

    # ── Tên file ─────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return (f"fmd_pinn_seed{self.seed}"
                f"_ls{int(self.use_levelset)}"
                f"_bs{int(self.use_bisection)}"
                f"_kn{int(self.use_knee)}"
                + (f"_{self.tag}" if self.tag else ""))

    # ── Vòng chính ───────────────────────────────────────────────────────────

    def run(self) -> dict:
        print(f"\n{'='*60}")
        print(f"FMD-PINN  seed={self.seed}  device={self.device}")
        print(f"  levelset={self.use_levelset}  bisection={self.use_bisection}"
              f"  knee={self.use_knee}")
        print(f"  oracle_budget={self.oracle_budget}")
        print(f"{'='*60}\n")

        call = self._resume_from

        # Pha mồi: LHS cho tới khi có cả điểm ổn định lẫn điểm sập
        if call == 0:
            seeds = lhs_region(N_SEED_CALLS, self.device)
            for i in range(N_SEED_CALLS):
                self._query(seeds[i], call, kind="seed")
                call += 1
            print(f"  [seed] {sum(1 for r in self.oracle_history if r['E_true'] > 0)}"
                  f"/{len(self.oracle_history)} sập")

        if not getattr(self, "_init_done", False):
            print("  [init] huấn luyện ban đầu ...")
            self._train(INIT_STEPS, log_every=500)
            self._init_done = True

        pbar = tqdm(total=self.oracle_budget, desc="Oracle calls", initial=call)
        while call < self.oracle_budget:
            if self.use_bisection and call % BISECT_EVERY == 0:
                p, kind = self._propose_bisect(), "bisect"
            elif self.use_levelset:
                p, kind = self._propose_levelset(), "levelset"
            else:
                p, kind = lhs_region(1, self.device)[0], "random"

            rec = self._query(p, call, kind=kind)
            self._train(STEPS_PER_CALL)

            call += 1
            pbar.update(1)
            pbar.set_postfix(E_true=f"{rec['E_true']:+.3f}",
                             E_pinn=f"{rec['E_pinn']:+.3f}", k=kind[:4])
            if call % CKPT_EVERY == 0 or call == self.oracle_budget:
                self._save_checkpoint(call)
        pbar.close()

        results = {
            "oracle_history": self.oracle_history,
            "seed": self.seed,
            "config": {"use_levelset": self.use_levelset,
                       "use_bisection": self.use_bisection,
                       "use_knee": self.use_knee,
                       "oracle_budget": self.oracle_budget},
        }
        self._save(results)
        return results

    # ── Gọi oracle ───────────────────────────────────────────────────────────

    def _query(self, p_hat: torch.Tensor, call: int, kind: str) -> dict:
        # Mọi lời gọi oracle bị giữ trong vùng cho phép — đây là
        # ràng buộc của phép thử ngoại suy, không phải của mô hình.
        p_hat = clamp_region(clamp_normalised(p_hat.detach()))
        alpha, beta, D = [x.item() for x in denormalise_params(p_hat)]
        t0 = time.time()
        r = solve_reaction_diffusion(alpha, beta, D)
        t_fem = time.time() - t0
        E_pinn = float(failure_functional(self.model, p_hat).item())
        rec = dict(oracle_call=call, alpha=alpha, beta=beta, D=D,
                   E_true=float(r["E"]), E_pinn=E_pinn,
                   t_fem=t_fem, kind=kind)
        self.oracle_history.append(rec)
        return rec

    # ── Đề xuất: tập mức không của PINN ──────────────────────────────────────

    def _build_scan_grid(self) -> torch.Tensor:
        """
        Lưới tham số để quét tập mức không, GIỚI HẠN trong vùng được phép gọi
        oracle. Ngoài vùng đó thì mô hình vẫn dự đoán được (ràng buộc PDE phủ
        toàn không gian), chỉ là không được hỏi oracle.
        """
        lo, hi = region_bounds(device=self.device)
        axes = []
        for i in range(3):
            a = float(lo[i]) + 0.02 * (float(hi[i]) - float(lo[i]))
            b = float(hi[i]) - 0.02 * (float(hi[i]) - float(lo[i]))
            axes.append(torch.linspace(a, b, SCAN_GRID, device=self.device))
        A, B, D = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([A.reshape(-1), B.reshape(-1), D.reshape(-1)], dim=1)

    def _scan(self, chunk: int = None) -> tuple:
        """
        Trả về (E, ||grad_p E||) trên lưới quét.

        Kích thước lô co theo EVAL_NT: mỗi điểm tham số kéo theo cả lưới đánh
        giá, và _scan giữ đồ thị để lấy gradient, nên bộ nhớ tỉ lệ với
        chunk * EVAL_NX^2 * EVAL_NT. Với EVAL_NT=24 thì chunk=512 cần gần 1.8
        triệu hàng một lượt, quá sức card 4 GB.
        """
        from src.sampling import eval_grid_size
        if chunk is None:
            chunk = max(16, int(getattr(cfg, "SCAN_ROWS", 400_000)
                                // max(eval_grid_size(), 1)))
        E_all, G_all = [], []
        self.model.eval()
        i = 0
        while i < len(self._scan_grid):
            p = self._scan_grid[i:i + chunk].clone().requires_grad_(True)
            try:
                E = failure_functional(self.model, p)
                g = torch.autograd.grad(E.sum(), p)[0]
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk <= 16:
                    raise
                chunk = max(16, chunk // 2)
                print(f"  [scan] hết VRAM, giảm lô xuống {chunk}")
                continue
            E_all.append(E.detach())
            G_all.append(g.norm(dim=1).detach())
            i += chunk
        self.model.train()
        return torch.cat(E_all), torch.cat(G_all)

    def _hist_points(self) -> torch.Tensor:
        if not self.oracle_history:
            return torch.empty(0, 3, device=self.device)
        a = torch.tensor([r["alpha"] for r in self.oracle_history],
                         device=self.device, dtype=torch.float32)
        b = torch.tensor([r["beta"] for r in self.oracle_history],
                         device=self.device, dtype=torch.float32)
        d = torch.tensor([r["D"] for r in self.oracle_history],
                         device=self.device, dtype=torch.float32)
        return normalise_params(a, b, d)

    def _level_offset(self) -> float:
        """Lệch mức b = trung bình (E_true − E_pinn) trên các lời gọi sát biên.
        PINN nắm đúng hình dạng nhưng trôi về mức; b kéo tập mức không về đúng
        chỗ trước khi chọn điểm gọi oracle. Không đổi huấn luyện, không đổi ∇E."""
        if not getattr(cfg, "LEVEL_CORRECT", False) or len(self.oracle_history) < 5:
            return 0.0
        E_true = torch.tensor([r["E_true"] for r in self.oracle_history],
                              device=self.device, dtype=torch.float32)
        near = E_true.abs() < getattr(cfg, "LEVEL_BAND", 0.3)
        if near.sum() < 3:
            return 0.0
        self.model.eval()
        with torch.no_grad():
            E_now = failure_functional(self.model, self._hist_points()[near])
        self.model.train()
        return float((E_true[near] - E_now).mean().item())

    def _propose_levelset(self) -> torch.Tensor:
        """
        Điểm trên tập mức không dự đoán, chọn theo:
          score = khoảng cách nhỏ nhất tới lịch sử
                + KNEE_WEIGHT * w(||grad_p E||)
        Số hạng hai ưu tiên theo độ dốc của biên; hướng ưu tiên do
        cfg.KNEE_PREFER quyết định (xem ghi chú bên dưới).
        """
        E, gnorm = self._scan()
        E = E + self._level_offset()
        band = BAND
        mask = E.abs() < band
        while mask.sum() < 10 and band < 1.0:
            band *= 2
            mask = E.abs() < band
        if mask.sum() == 0:
            return lhs_region(1, self.device)[0]

        cand = self._scan_grid[mask]
        hist = self._hist_points()
        if len(hist) > 0:
            d_min = torch.cdist(cand, hist).min(dim=1).values
        else:
            d_min = torch.ones(len(cand), device=self.device)

        score = d_min
        if self.use_knee:
            # Ưu tiên theo độ dốc của biên. Dọc theo biên giải tích
            # alpha = 2*pi^2*D + beta*u_thr^2, chuẩn gradient chuẩn hoá là 9.60
            # ở beta=0.6 và 1.44 ở beta=4.0: vùng knee là chỗ gradient LỚN nhất,
            # không phải nhỏ nhất. Bản đầu dùng 1/(1+r), tức thưởng cho gradient
            # nhỏ, nên nó chủ động TRÁNH vùng knee; số liệu khớp: biến thể đó có
            # δ_H vùng knee 0.0778 (tệ nhất) và báo động giả knee 42.1%.
            # "steep": thưởng cho gradient lớn, vì biên ở đó nhạy nên cần thêm
            #          lời gọi oracle. "flat": giữ hành vi cũ để đối chứng.
            gc = gnorm[mask]
            r = gc / (gc.median() + 1e-8)
            if getattr(cfg, "KNEE_PREFER", "steep") == "steep":
                w = r / (1.0 + r)
            else:
                w = 1.0 / (1.0 + r)
            score = score + KNEE_WEIGHT * w

        p = cand[score.argmax()]
        # nhiễu nhỏ để không lặp lại đúng nút lưới, rồi kéo về đúng tập mức không
        p = clamp_normalised(p + torch.randn(3, device=self.device) * 0.01)
        return self._project_to_levelset(p)

    def _project_to_levelset(self, p: torch.Tensor) -> torch.Tensor:
        """Newton trên E_pinn: p <- p - E * gradE / |gradE|^2.
        score đã chọn VỊ TRÍ DỌC theo biên; bước này sửa KHOẢNG CÁCH tới biên."""
        n_iter = getattr(cfg, "PROJECT_ITERS", 3)
        if n_iter <= 0:
            return p
        cap = getattr(cfg, "PROJECT_MAX_STEP", 0.05)
        self.model.eval()
        for _ in range(n_iter):
            q = p.detach().clone().unsqueeze(0).requires_grad_(True)
            E = failure_functional(self.model, q)
            g = torch.autograd.grad(E.sum(), q)[0][0]
            step = (E.detach()[0] * g / (g.pow(2).sum() + 1e-8)).clamp(-cap, cap)
            p = clamp_normalised(p - step)
        self.model.train()
        return p.detach()

    # ── Đề xuất: bisection trên oracle ───────────────────────────────────────

    def _propose_bisect(self) -> torch.Tensor:
        """
        Trung điểm của cặp bracket (E_true<0, E_true>0) gần nhau nhất mà
        trung điểm còn xa lịch sử. Không dùng gradient của PINN, nên nó sửa
        được chỗ PINN lệch thay vì nhân nó lên.
        """
        hist = self._hist_points()
        E_true = torch.tensor([r["E_true"] for r in self.oracle_history],
                              device=self.device, dtype=torch.float32)
        s_idx = (E_true < 0).nonzero(as_tuple=True)[0]
        c_idx = (E_true > 0).nonzero(as_tuple=True)[0]
        if len(s_idx) == 0 or len(c_idx) == 0:
            return self._propose_levelset() if self.use_levelset \
                else lhs_region(1, self.device)[0]

        S, C = hist[s_idx], hist[c_idx]
        pair_d = torch.cdist(S, C)                      # (ns, nc)
        mid = 0.5 * (S.unsqueeze(1) + C.unsqueeze(0))   # (ns, nc, 3)
        mid_flat = mid.reshape(-1, 3)
        d_hist = torch.cdist(mid_flat, hist).min(dim=1).values
        # cặp càng gần nhau càng đã hội tụ; ưu tiên trung điểm mới, bracket hẹp
        score = d_hist - 0.5 * pair_d.reshape(-1)
        return clamp_normalised(mid_flat[score.argmax()])

    # ── Huấn luyện ───────────────────────────────────────────────────────────

    def _param_batch(self, hist: torch.Tensor = None,
                     E_abs: torch.Tensor = None) -> torch.Tensor:
        """
        Một nửa LHS phủ toàn không gian, một nửa quanh các điểm biên đã biết
        (|E_true| nhỏ). Không có replay anchor chọn tay.

        hist/E_abs được truyền từ _train để không phải dựng lại tensor lịch sử
        ở mỗi bước — đó là chi phí O(H) mỗi bước, nên thời gian một lời gọi
        tăng dần theo số lời gọi (2.0 s lúc đầu lên 4.8 s ở cuối).
        """
        n_half = N_PARAM_BATCH // 2
        p = lhs(N_PARAM_BATCH - n_half, 3, self.device)
        if hist is not None and len(hist) > 0:
            k = max(min(n_half, len(hist)), 1)
            near = hist[E_abs.argsort()[:k]]
            idx = torch.randint(0, len(near), (n_half,), device=self.device)
            jitter = near[idx] + torch.randn(n_half, 3, device=self.device) * 0.03
            p = torch.cat([p, clamp_normalised(jitter)], dim=0)
        else:
            p = torch.cat([p, lhs(n_half, 3, self.device)], dim=0)
        return p

    def _train(self, n_steps: int, log_every: int = 0):
        self.model.train()
        hist = self._hist_points()
        E_true_all = torch.tensor([r["E_true"] for r in self.oracle_history],
                                  device=self.device, dtype=torch.float32)
        E_abs = E_true_all.abs()

        for step in range(n_steps):
            p_batch = self._param_batch(hist, E_abs)            # (P,3)
            P = p_batch.shape[0]

            # Mỗi p có bộ điểm collocation riêng, ghép thành một lượt forward
            xyt = collocation(P * N_COLL_PER_P, self.device)
            p_pde = p_batch.repeat_interleave(N_COLL_PER_P, dim=0)

            n_bc = max(4, cfg.N_BOUNDARY // 4) * 4
            xyt_bc, normals = boundary_points(n_bc, self.device)
            p_bc = p_batch[torch.randint(0, P, (xyt_bc.shape[0],),
                                         device=self.device)]

            xy_ic = initial_points(cfg.N_INITIAL, self.device)
            p_ic = p_batch[torch.randint(0, P, (xy_ic.shape[0],),
                                         device=self.device)]

            # Từng thành phần tính với p tương ứng của điểm đó
            loss = (loss_pde(self.model, xyt, p_pde)
                    + cfg.LAMBDA_BC * loss_bc(self.model, xyt_bc, p_bc, normals)
                    + cfg.LAMBDA_IC * loss_ic(self.model, xy_ic, p_ic))

            if len(hist) > 0:
                k = min(N_SUP_BATCH, len(hist))
                sel = torch.randperm(len(hist), device=self.device)[:k]
                loss = loss + LAMBDA_SUP * loss_supervised(
                    self.model, hist[sel], E_true_all[sel])

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.GRAD_CLIP)
            self.opt.step()

            if log_every and step % log_every == 0:
                print(f"    step={step:5d}/{n_steps}  loss={loss.item():.4e}")

    # ── Lưu / nạp ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, call: int):
        torch.save(self.model.state_dict(),
                   os.path.join(cfg.CKPT_DIR, f"{self.name}_call{call}.pt"))
        with open(os.path.join(cfg.RESULTS_DIR,
                               f"{self.name}_history_partial.json"), "w") as f:
            json.dump({"oracle_history": self.oracle_history,
                       "oracle_call": call,
                       "init_steps": INIT_STEPS}, f, default=float)

    def _save(self, results: dict):
        torch.save(self.model.state_dict(),
                   os.path.join(cfg.CKPT_DIR, f"{self.name}.pt"))
        path = os.path.join(cfg.RESULTS_DIR, f"{self.name}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n✓ Kết quả đã lưu: {path}")

    def load_checkpoint(self) -> bool:
        """
        Nạp lại phiên dở. Khác bản cũ: nếu không nạp được .pt thì XOÁ luôn
        lịch sử trong bộ nhớ, để không bao giờ có chuyện mô hình mới tinh
        chạy tiếp trên lịch sử cũ (lỗi --force-restart của run_04).
        """
        hp = os.path.join(cfg.RESULTS_DIR, f"{self.name}_history_partial.json")
        if not os.path.exists(hp):
            return False
        saved = json.load(open(hp))
        call = int(saved["oracle_call"])
        ckpt = os.path.join(cfg.CKPT_DIR, f"{self.name}_call{call}.pt")
        if not os.path.exists(ckpt):
            print("[RESUME] thiếu .pt — bắt đầu lại từ đầu.")
            return False
        try:
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        except RuntimeError as e:
            print(f"[RESUME] checkpoint không khớp — bắt đầu lại. ({e})")
            return False
        seen, dedup = set(), []
        for r in saved["oracle_history"]:
            if r["oracle_call"] not in seen:
                seen.add(r["oracle_call"])
                dedup.append(r)
        self.oracle_history = dedup
        self._resume_from = call
        # Checkpoint sinh ra từ một cấu hình huấn luyện nhỏ hơn (ví dụ
        # smoketest) thì không đủ tư cách bỏ qua pha huấn luyện ban đầu.
        self._init_done = int(saved.get("init_steps", 0)) >= INIT_STEPS
        if not self._init_done:
            print("[RESUME] checkpoint từ cấu hình nhỏ hơn — sẽ huấn luyện "
                  "ban đầu lại trước khi tiếp tục.")
        print(f"[RESUME] tiếp từ call={call}, {len(dedup)} bản ghi")
        return True

    @staticmethod
    def clear_state(name: str):
        """Xoá sạch history + checkpoint của một cấu hình. Dùng cho --force-restart."""
        for p in [os.path.join(cfg.RESULTS_DIR, f"{name}_history_partial.json"),
                  os.path.join(cfg.RESULTS_DIR, f"{name}.json"),
                  os.path.join(cfg.CKPT_DIR, f"{name}.pt")]:
            if os.path.exists(p):
                os.remove(p)
        for f in os.listdir(cfg.CKPT_DIR):
            if f.startswith(f"{name}_call") and f.endswith(".pt"):
                os.remove(os.path.join(cfg.CKPT_DIR, f))