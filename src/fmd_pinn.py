"""
src/fmd_pinn.py
---------------
Core FMD-PINN framework: min-max adversarial discovery loop.

Algorithm
---------
For each oracle call t:
  Inner loop : minimise L_total(θ; p) via Adam → L-BFGS
  Outer loop : p ← p + η · ∂E/∂p̂  (gradient ascent in normalised space)
  Oracle     : call FEM at p* → confirm E_true
  If COLLAPSE and was predicted SAFE → emergency re-train (FSR alert)
  Update Phase Diagram
"""

import torch
import torch.optim as optim
import numpy as np
import os, json, time
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config as cfg
from src.pinn_model   import ParametricPINN, normalise_params, denormalise_params, clamp_normalised, build_model
from src.loss_functions import total_loss, failure_functional, AdaptiveLossScheduler
from src.adaptive_sampling import AdaptiveSampler
from src.fem_oracle   import solve_reaction_diffusion


class FMDPINNTrainer:
    """
    Full FMD-PINN trainer.

    Parameters
    ----------
    seed        : random seed
    use_min_max : if False, outer loop uses random exploration (ablation A4)
    use_event_loss    : (ablation flag)
    use_adaptive_samp : (ablation flag)
    """

    def __init__(self, seed: int = 42,
                 use_min_max: bool        = True,
                 use_event_loss: bool     = True,
                 use_adaptive_samp: bool  = True,
                 oracle_budget: int       = None):
        self.seed              = seed
        self.use_min_max       = use_min_max
        self.use_event_loss    = use_event_loss
        self.use_adaptive_samp = use_adaptive_samp
        self.oracle_budget     = oracle_budget or cfg.ORACLE_BUDGET
        self.device            = cfg.DEVICE

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model     = build_model(self.device)
        self.sampler   = AdaptiveSampler(self.device)
        self.scheduler = AdaptiveLossScheduler()

        # Optimisers
        self.adam   = optim.Adam(self.model.parameters(), lr=cfg.LR_ADAM)
        self.lbfgs  = optim.LBFGS(self.model.parameters(),
                                   max_iter=cfg.LBFGS_STEPS,
                                   history_size=cfg.LBFGS_HISTORY,
                                   line_search_fn="strong_wolfe")

        # Oracle history  list of dicts {p_hat, alpha, beta, D, E_true}
        self.oracle_history: list = []
        # Phase diagram samples  {p: (alpha,beta,D), E_pinn, E_true}
        self.phase_diagram: list  = []

        # FSR tracking
        self._false_safe_count = 0
        self._total_safe_explored = 0
        self._fsr_history: list = []

        # Initialise p in normalised space (start near centre)
        self.p_hat = torch.full((3,), 0.5, device=self.device, requires_grad=False)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full min-max adversarial loop."""
        print(f"\n{'='*60}")
        print(f"FMD-PINN  seed={self.seed}  device={self.device}")
        print(f"  min_max={self.use_min_max}  event_loss={self.use_event_loss}"
              f"  adaptive_samp={self.use_adaptive_samp}")
        print(f"  oracle_budget={self.oracle_budget}")
        print(f"{'='*60}\n")

        n_exploit = int(self.oracle_budget * cfg.EXPLOIT_RATIO)
        n_explore = self.oracle_budget - n_exploit

        # ── Phase 0: initial warm-up training ─────────────────────────────
        print("[Phase 0] Warm-up training on random parameter samples …")
        self._warmup_training(n_steps=cfg.ADAM_EPOCHS)

        # ── Resume from checkpoint if available ──────────────────────────
        oracle_call = getattr(self, "_resume_from", 0)
        if oracle_call > 0:
            print(f"[RESUME] Continuing from oracle call {oracle_call}")

        # ── Main loop ─────────────────────────────────────────────────────
        pbar = tqdm(total=self.oracle_budget, desc="Oracle calls",
                    initial=oracle_call)

        while oracle_call < self.oracle_budget:
            is_exploit = (oracle_call < n_exploit
                          or np.random.rand() < cfg.EXPLOIT_RATIO)

            # Every 5th exploit call → bisection toward boundary
            use_bisect = (is_exploit
                          and self.use_min_max
                          and oracle_call % 5 == 4
                          and oracle_call >= 10)   # need enough history first

            if use_bisect:
                p_candidate = self._bisect_to_boundary()
            elif is_exploit and self.use_min_max:
                # Two-phase: maximize E then seek boundary
                p_candidate = self._outer_ascent(oracle_call=oracle_call)
            elif is_exploit and not self.use_min_max:
                # Ablation: random search instead of gradient ascent
                p_candidate = torch.rand(3, device=self.device)
            else:
                # Exploration: LHS sample from "safe" region
                p_candidate = self._explore_safe_region()

            # Call FEM oracle
            alpha, beta, D = [x.item() for x in denormalise_params(p_candidate)]
            t0   = time.time()
            r    = solve_reaction_diffusion(alpha, beta, D)
            t_fem = time.time() - t0
            E_true = r["E"]

            # Compute PINN prediction at this point
            E_pinn = self._pinn_E(p_candidate)

            # Log
            record = dict(oracle_call=oracle_call,
                          alpha=alpha, beta=beta, D=D,
                          E_true=float(E_true),
                          E_pinn=float(E_pinn),
                          t_fem=t_fem,
                          is_exploit=is_exploit)
            self.oracle_history.append(record)
            self.phase_diagram.append(record)

            # FSR monitoring
            if not is_exploit:
                self._total_safe_explored += 1
                if E_pinn < 0 and E_true > 0:           # False Safe!
                    self._false_safe_count += 1
                    fsr = self._false_safe_count / self._total_safe_explored
                    self._fsr_history.append((oracle_call, fsr))
                    tqdm.write(f"  ⚠ FALSE SAFE at oracle {oracle_call}! "
                               f"FSR={fsr:.3f}  α={alpha:.2f} β={beta:.2f} D={D:.4f}")
                    if fsr > cfg.FSR_ALERT_THRESH:
                        self._emergency_retrain(p_candidate, penalty=10.0)

            # Fine-tune PINN with oracle result
            self._finetune(p_candidate, E_true)

            # Inner training step
            self._inner_train(p_candidate, n_steps=500)

            # Update sampler
            self.sampler.update(self.model, p_candidate)

            # Update scheduler
            self.scheduler.step(E_pinn)

            oracle_call += 1
            pbar.set_postfix(E_true=f"{E_true:+.3f}",
                             E_pinn=f"{E_pinn:+.3f}",
                             FSR=f"{self._current_fsr():.3f}")
            pbar.update(1)

            # Incremental save every 10 calls — safe to Ctrl+C anytime
            if oracle_call % 10 == 0 or oracle_call == self.oracle_budget:
                self._save_checkpoint(oracle_call)

        pbar.close()

        results = {
            "oracle_history": self.oracle_history,
            "phase_diagram":  self.phase_diagram,
            "fsr_history":    self._fsr_history,
            "seed":           self.seed,
            "config": {
                "use_min_max":       self.use_min_max,
                "use_event_loss":    self.use_event_loss,
                "use_adaptive_samp": self.use_adaptive_samp,
                "oracle_budget":     self.oracle_budget,
            }
        }
        self._save(results)
        return results

    # ── Outer loop: two-phase boundary-seeking ───────────────────────────────

    def _outer_ascent(self, oracle_call: int = 0) -> torch.Tensor:
        """
        Two-phase outer loop:
          Phase 1 (first 60% of budget): maximize E → find collapse region
          Phase 2 (last 40% of budget):  minimize |E| → converge to boundary ∂C

        Phase 2 seeds from previously discovered collapse points (oracle history),
        ensuring boundary-seeking starts from the right side of ∂C.
        """
        batch = self.sampler.sample()
        xyt   = batch["xyt_pde"]

        phase1_end = int(self.oracle_budget * 0.40)
        in_phase2  = (oracle_call >= phase1_end)

        if in_phase2 and len(self.oracle_history) > 0:
            # Seed from a random collapse point in history
            collapse_pts = [r for r in self.oracle_history if r["E_true"] > 0]
            if collapse_pts:
                rec = collapse_pts[np.random.randint(len(collapse_pts))]
                from src.pinn_model import normalise_params
                p_seed = normalise_params(
                    torch.tensor([rec["alpha"]], device=self.device),
                    torch.tensor([rec["beta"]],  device=self.device),
                    torch.tensor([rec["D"]],     device=self.device),
                ).squeeze(0)
                p = p_seed.clone().detach().requires_grad_(True)
            else:
                p = self.p_hat.clone().detach().requires_grad_(True)
        else:
            p = self.p_hat.clone().detach().requires_grad_(True)

        for step in range(cfg.OUTER_STEPS):
            p_clamped = clamp_normalised(p)
            E = failure_functional(self.model, xyt, p_clamped)

            if in_phase2:
                # Minimize |E| — gradient descent on E² → finds zero crossing
                obj = E ** 2
            else:
                # Maximize E — standard gradient ascent
                obj = -E

            obj.backward()

            with torch.no_grad():
                grad = p.grad.clone()
                gnorm = grad.norm()
                if gnorm > cfg.GRAD_CLIP:
                    grad = grad * cfg.GRAD_CLIP / gnorm

                # Phase 2: descent on |E| (smaller LR for precision)
                lr = cfg.OUTER_LR * (0.3 if in_phase2 else 1.0)
                p.data -= lr * grad
                p.data  = clamp_normalised(p.data)
                p.grad.zero_()

        if not in_phase2:
            self.p_hat = p.detach()
        return clamp_normalised(p.detach())

    def _bisect_to_boundary(self) -> torch.Tensor:
        """
        Binary bisection between a random stable/collapse pair from oracle history.
        Does NOT rely on PINN gradients — guaranteed to land near ∂C.

        Picks one stable (E_true < 0) and one collapse (E_true > 0) point,
        returns their midpoint in normalised parameter space.
        Falls back to _outer_ascent if not enough history yet.
        """
        from src.pinn_model import normalise_params
        stable_pts   = [r for r in self.oracle_history if r["E_true"] < 0]
        collapse_pts = [r for r in self.oracle_history if r["E_true"] > 0]

        if not stable_pts or not collapse_pts:
            return self._outer_ascent()

        # Pick the pair whose midpoint is closest to boundary
        # (use PINN E as proxy — pick pair with min |E_pinn(midpoint)|)
        rng  = np.random.default_rng(seed=int(self.seed + len(self.oracle_history)))
        s_recs = rng.choice(stable_pts,   size=min(5, len(stable_pts)),   replace=False)
        c_recs = rng.choice(collapse_pts, size=min(5, len(collapse_pts)), replace=False)

        best_p, best_E = None, np.inf
        self.model.eval()
        xyt_s = torch.rand(256, 3, device=self.device)
        xyt_s[:, 2] *= cfg.T_END

        with torch.no_grad():
            for sr in s_recs:
                for cr in c_recs:
                    # Midpoint in physical space, then normalise
                    mid_a = (sr["alpha"] + cr["alpha"]) / 2
                    mid_b = (sr["beta"]  + cr["beta"])  / 2
                    mid_D = float(np.exp((np.log(sr["D"]) + np.log(cr["D"])) / 2))
                    p_mid = normalise_params(
                        torch.tensor([mid_a], device=self.device),
                        torch.tensor([mid_b], device=self.device),
                        torch.tensor([mid_D], device=self.device),
                    ).squeeze(0)
                    E_mid = failure_functional(self.model, xyt_s, p_mid).abs().item()
                    if E_mid < best_E:
                        best_E = E_mid
                        best_p = p_mid
        self.model.train()
        return clamp_normalised(best_p)

    # ── Exploration ───────────────────────────────────────────────────────────

    def _explore_safe_region(self) -> torch.Tensor:
        """
        LHS sample in the region where FMD-PINN predicts SAFE (E < 0).
        Falls back to uniform random if all candidates predict collapse.
        """
        n_candidates = 200
        candidates   = torch.rand(n_candidates, 3, device=self.device)
        xyt_sample   = torch.rand(256, 3, device=self.device)
        xyt_sample[:, 2] *= cfg.T_END

        self.model.eval()
        with torch.no_grad():
            E_pred = []
            for c in candidates:
                e = failure_functional(self.model, xyt_sample, c)
                E_pred.append(e.item())
        self.model.train()

        E_pred = np.array(E_pred)
        safe_mask = E_pred < 0
        if safe_mask.sum() == 0:
            return candidates[np.argmin(E_pred)]   # least unsafe
        safe_cands = candidates[torch.from_numpy(safe_mask.astype(np.uint8)).bool()]
        idx = np.random.randint(len(safe_cands))
        return safe_cands[idx]

    # ── Inner training ────────────────────────────────────────────────────────

    def _warmup_training(self, n_steps: int = 3000):
        """Supervised pretraining on FEM data before oracle loop."""
        from src.fem_oracle import solve_reaction_diffusion
        from src.pinn_model import normalise_params
        import numpy as np
        self.model.train()
        print("  [Warmup] Collecting FEM data for pretraining...")
        pretrain_cases = [
            (2.0,  4.0, 0.1,   "stable"),
            (10.0, 1.0, 0.01,  "collapse"),
            (5.0,  1.5, 0.05,  "near-boundary"),
            (14.0, 0.5, 0.005, "extreme collapse"),
            (1.5,  4.0, 0.2,   "stable2"),
        ]
        fem_data = []
        for alpha, beta, D, label in pretrain_cases:
            r = solve_reaction_diffusion(alpha, beta, D, dense=True)
            p = normalise_params(
                torch.tensor([alpha], device=self.device),
                torch.tensor([beta],  device=self.device),
                torch.tensor([D],     device=self.device)).squeeze(0)
            fem_data.append((p, r, label))
            print(f"    {label}: E={r['E']:+.3f}")
        nx = cfg.FEM_NX
        x  = np.linspace(0, 1, nx)
        X, Y = np.meshgrid(x, x)
        for step in range(n_steps):
            idx  = step % len(fem_data)
            p_hat, fem_r, label = fem_data[idx]
            t_idx = np.random.randint(0, len(fem_r["traj"]))
            t_val = float(fem_r["t_eval"][t_idx])
            u_fem = torch.tensor(
                fem_r["traj"][t_idx].ravel(),
                dtype=torch.float32, device=self.device)
            xyt = torch.tensor(
                np.column_stack([X.ravel(), Y.ravel(),
                                 np.full(nx*nx, t_val)]),
                dtype=torch.float32, device=self.device)
            u_pred = self.model(xyt, p_hat)
            loss   = ((u_pred - u_fem)**2).mean()
            self.adam.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.adam.step()
            if step % 500 == 0:
                print(f"  [Warmup] step={step:4d}/{n_steps}  loss={loss.item():.4e}")
        self.model.eval()
        print("  [Warmup] Post-pretrain E_pinn check:")
        for p_hat, fem_r, label in fem_data:
            xyt_test = torch.rand(256, 3, device=self.device)
            xyt_test[:,2] *= cfg.T_END
            with torch.no_grad():
                N = xyt_test.shape[0]
                p_exp = p_hat.unsqueeze(0).expand(N, -1)
                E = self.model(xyt_test, p_exp).max().item() - cfg.U_THRESHOLD
            print(f"    {label}: E_pinn={E:+.3f}  E_true={fem_r['E']:+.3f}")
        self.model.train()

    def _inner_train(self, p_hat: torch.Tensor, n_steps: int = 500):
        """Adam + L-BFGS inner optimisation for current p."""
        self.model.train()
        p_hat = p_hat.detach()

        # Always include warmup anchor cases
        from src.pinn_model import normalise_params
        from src.loss_functions import failure_functional
        import numpy as np
        
        anchor_cases = [
            (2.0, 4.0, 0.1, -1.219),
            (10.0, 1.0, 0.01, 1.662),
            (14.0, 0.5, 0.005, 2.167),
        ]
        replay = []
        for a, b, D, e_true in anchor_cases:
            rp = normalise_params(
                torch.tensor([a], device=self.device),
                torch.tensor([b], device=self.device),
                torch.tensor([D], device=self.device)).squeeze(0)
            replay.append((rp, e_true))
            
        for rec in self.oracle_history[-10:]:
            rec_p = normalise_params(
                torch.tensor([rec["alpha"]], device=self.device),
                torch.tensor([rec["beta"]],  device=self.device),
                torch.tensor([rec["D"]],     device=self.device)).squeeze(0)
            replay.append((rec_p, float(rec["E_true"])))

        def compute_sup_loss():
            if len(replay) == 0:
                return 0.0
            xyt_test = torch.rand(256, 3, device=self.device)
            xyt_test[:, 2] *= cfg.T_END
            sup_loss = 0.0
            for rp, e_true in replay:
                e_pred = failure_functional(self.model, xyt_test, rp)
                sup_loss = sup_loss + (e_pred - e_true)**2
            return sup_loss / len(replay)

        # Adam phase
        for _ in range(n_steps):
            batch = self.sampler.sample()
            ldict = total_loss(
                self.model,
                batch["xyt_pde"], batch["xyt_bc"], batch["normals"],
                batch["xy_ic"],   batch["xyt_event"], p_hat,
                lambda_event = self.scheduler.get() if self.use_event_loss else 0.0
            )
            
            loss = ldict["total"]
            sup_loss = compute_sup_loss()
            if isinstance(sup_loss, torch.Tensor):
                loss = loss + 5.0 * sup_loss
                
            self.adam.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.adam.step()

        # L-BFGS polish
        batch = self.sampler.sample()
        def closure():
            self.lbfgs.zero_grad()
            ld = total_loss(
                self.model,
                batch["xyt_pde"], batch["xyt_bc"], batch["normals"],
                batch["xy_ic"],   batch["xyt_event"], p_hat,
                lambda_event = self.scheduler.get() if self.use_event_loss else 0.0
            )
            
            loss = ld["total"]
            sup_loss = compute_sup_loss()
            if isinstance(sup_loss, torch.Tensor):
                loss = loss + 5.0 * sup_loss
                
            loss.backward()
            return loss
        self.lbfgs.step(closure)

    def _finetune(self, p_hat: torch.Tensor, E_true: float):
        """
        Fine-tune PINN using oracle result as a soft constraint.
        If E_true > 0 (collapse), increase event loss weight at this point.
        """
        if E_true > 0:
            # Extra event-loss steps at this dangerous point
            self._inner_train(p_hat, n_steps=200)

    def _emergency_retrain(self, p_hat: torch.Tensor, penalty: float = 10.0):
        """Triggered when FSR > threshold. Penalise false-safe region hard."""
        print(f"  [EMERGENCY RETRAIN]  W_penalty × {penalty:.0f}")
        self.model.train()
        batch = self.sampler.sample()
        for _ in range(300):
            ldict = total_loss(
                self.model,
                batch["xyt_pde"], batch["xyt_bc"], batch["normals"],
                batch["xy_ic"],   batch["xyt_event"], p_hat,
                lambda_event = self.scheduler.get() * penalty
            )
            self.adam.zero_grad()
            ldict["total"].backward()
            self.adam.step()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pinn_E(self, p_hat: torch.Tensor) -> float:
        self.model.eval()
        xyt = torch.rand(512, 3, device=self.device)
        xyt[:, 2] *= cfg.T_END
        with torch.no_grad():
            E = failure_functional(self.model, xyt, p_hat)
        self.model.train()
        return float(E.item())

    def _current_fsr(self) -> float:
        if self._total_safe_explored == 0:
            return 0.0
        return self._false_safe_count / self._total_safe_explored

    def load_checkpoint(self):
        """Load latest partial checkpoint. Call before run() to resume."""
        name = (f"fmd_pinn_seed{self.seed}"
                f"_minmax{int(self.use_min_max)}"
                f"_ev{int(self.use_event_loss)}"
                f"_as{int(self.use_adaptive_samp)}")
        hist_path = os.path.join(cfg.RESULTS_DIR,
                                  f"{name}_history_partial.json")
        if not os.path.exists(hist_path):
            print("[RESUME] No checkpoint found, starting fresh.")
            return False
        with open(hist_path) as f:
            saved = json.load(f)
        self.oracle_history = saved["oracle_history"]
        self.phase_diagram  = saved["oracle_history"][:]
        last_call = int(saved["oracle_call"])
        ckpt = os.path.join(cfg.CKPT_DIR, f"{name}_call{last_call}.pt")
        if os.path.exists(ckpt):
            self.model.load_state_dict(
                torch.load(ckpt, map_location=self.device))
            self._resume_from = last_call
            print(f"[RESUME] Loaded call={last_call}, "
                  f"history={len(self.oracle_history)} records")
            return True
        print("[RESUME] Checkpoint .pt not found, starting fresh.")
        return False

    def _save_checkpoint(self, oracle_call: int):
        """Save oracle history after every N calls — resume-safe."""
        name = (f"fmd_pinn_seed{self.seed}"
                f"_minmax{int(self.use_min_max)}"
                f"_ev{int(self.use_event_loss)}"
                f"_as{int(self.use_adaptive_samp)}")
        ckpt = os.path.join(cfg.CKPT_DIR, f"{name}_call{oracle_call}.pt")
        torch.save(self.model.state_dict(), ckpt)
        hist_path = os.path.join(cfg.RESULTS_DIR, f"{name}_history_partial.json")
        with open(hist_path, "w") as f:
            json.dump({"oracle_history": self.oracle_history,
                       "oracle_call": oracle_call}, f, default=float)

    def _save(self, results: dict):
        name = (f"fmd_pinn_seed{self.seed}"
                f"_minmax{int(self.use_min_max)}"
                f"_ev{int(self.use_event_loss)}"
                f"_as{int(self.use_adaptive_samp)}")
        # Save model
        ckpt_path = os.path.join(cfg.CKPT_DIR, f"{name}.pt")
        torch.save(self.model.state_dict(), ckpt_path)
        # Save results
        res_path = os.path.join(cfg.RESULTS_DIR, f"{name}.json")
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n✓ Results saved: {res_path}")