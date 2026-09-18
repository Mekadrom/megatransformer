"""MuonClip / qk-clip (Kimi K2): keep attention logits bounded by rescaling W_q / W_k.

WHY THIS AND NOT `attn_logit_cap`. Both bound attention logits, in different places:

  attn_logit_cap   forward pass, every step: scores = cap * tanh(scores / cap). Changes the
                   function the model computes, and costs the SDPA fast path
                   (transformer.py:271 gates `use_sdpa` on the cap being None), which
                   profiling put at a large share of world-model CUDA time.
  qk-clip (here)   optimizer side, after the step: if an OBSERVED per-head max logit exceeds
                   tau, shrink that head's W_q and W_k blocks. The forward pass is untouched,
                   SDPA is kept, and on steps where nothing exceeds tau this is a no-op.

WHY IT IS A MUON CONCERN SPECIFICALLY. Muon orthogonalizes its update, so every singular
direction of a weight matrix gets the same step size. That is the point -- it is why Muon
conditions well -- but it also means spectral norms grow more freely than under Adam, and
sigma(W_q)*sigma(W_k) is exactly what sets the attention logit scale.

THE RULE (per head h):

    gamma_h = min(1, tau / S_h)          S_h = observed max pre-softmax logit for head h
    W_q[h] *= gamma_h ** alpha
    W_k[h] *= gamma_h ** (1 - alpha)     alpha = 0.5 splits it evenly

Applied only when S_h > tau, so gamma_h < 1. Since the logit is bilinear in (W_q, W_k), the
product of the two scalings is exactly gamma_h -- one factor of tau/S_h.

⚠️ GQA. W_k has n_query_groups head blocks, not n_heads, and one k head serves n_rep q heads.
Scaling a k block affects every q head in its group, so the group takes the MINIMUM gamma
(i.e. the most aggressive clip any of its members asked for). The q blocks still get their own
gamma. This is conservative: a head can end up clipped slightly harder than it asked for,
never softer.

RECURRENT TRUNK. Three things follow from the trunk calling ONE attention module many times
per forward pass:

  1. The per-module max must span iterations, which is why `_qk_max_logit` accumulates with
     torch.maximum inside an armed step -- and why it is reset at arm time, not never.
  2. `named_modules()` de-duplicates shared modules, so a weight-shared block is discovered
     and clipped ONCE. Clipping it per-iteration would compound gamma**n_iters and collapse
     the block.
  3. The observed max is depth-dependent: the Poisson-sampled iteration count varies per step
     (see recurrent.py:273 n_k_steps), so a 60-iteration step samples more attention maps than
     an 8-iteration one and reports a higher max for the same weights. That makes the measured
     series noisy in a way a non-recurrent model's is not. Prefer a tau set from a HIGH
     percentile of observed maxima over several probes, not from one reading, and expect the
     post-clip logit to land NEAR tau rather than exactly on it -- clipping early iterations
     changes the activations the later ones see, so one application is a control step, not an
     exact solve.

SAMPLING. The probe materializes the (N, heads, t, T) score matrix that SDPA exists to avoid,
so it runs every `probe_every` steps rather than continuously. Logit scale moves slowly --
the AdamW baseline's weight-side bound moved ~0.5-0.7% per 1000 steps -- so sampling costs
almost nothing in responsiveness. Between probes the last observed value is carried, and the
clip still applies, because a head that was over tau does not drop back under it by itself.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class QKClipController:
    """Arms the forward probe and applies qk-clip after optimizer steps.

    Usage: build once with the model, call `maybe_arm(step)` BEFORE the forward pass and
    `apply(step)` AFTER `optimizer.step()`.
    """

    def __init__(self, model: nn.Module, tau: float, probe_every: int = 50,
                 alpha: float = 0.5, verbose: bool = False):
        if tau is None or tau <= 0:
            raise ValueError(f"qk_clip tau must be positive, got {tau}")
        self.tau = float(tau)
        self.probe_every = max(int(probe_every), 1)
        self.alpha = float(alpha)
        self.verbose = verbose
        self.modules: List[Tuple[str, nn.Module]] = []
        for name, m in model.named_modules():
            # An attention module for our purposes is one that has the projections we would
            # rescale AND the head geometry we need to slice them. Anything else is skipped
            # rather than guessed at.
            if (hasattr(m, "q_proj") and hasattr(m, "k_proj")
                    and hasattr(m, "n_heads") and hasattr(m, "d_queries")
                    and isinstance(getattr(m, "q_proj", None), nn.Linear)
                    and isinstance(getattr(m, "k_proj", None), nn.Linear)):
                if not m.q_proj.weight.requires_grad:
                    continue          # frozen spine (SmolLM2) -- nothing to clip
                self.modules.append((name, m))
        self.armed = False
        self.last_stats: Dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.modules)

    def maybe_arm(self, step: int) -> None:
        """Turn the probe on for this step's forward pass, if it is a sampled step.

        Arming RESETS the accumulator. `_qk_max_logit` is a running max, and inside one armed
        step that is exactly right -- the recurrent trunk calls the same attention module once
        per iteration, so the max must span all of them. Across steps it is wrong: without the
        reset the value would be a max over all history, so one early spike would keep the
        clip firing forever against a number the weights no longer produce.
        """
        want = (step % self.probe_every) == 0
        if want == self.armed:
            return
        for _, m in self.modules:
            m._qk_probe_active = want
            if want:
                m._qk_max_logit = None
        self.armed = want

    def disarm(self) -> None:
        """Turn the probe off. Called at the end of every apply() so the flag is live for
        exactly one step's forward passes and never leaks into an eval pass, whose batch
        would otherwise contribute to the measured max."""
        if not self.armed:
            return
        for _, m in self.modules:
            m._qk_probe_active = False
        self.armed = False

    @torch.no_grad()
    def apply(self, step: int) -> Dict[str, float]:
        """Rescale W_q / W_k for every head whose observed max logit exceeds tau."""
        was_armed = self.armed
        n_clipped = 0
        n_heads_total = 0
        max_seen = float("-inf")
        min_gamma = 1.0
        for name, m in self.modules:
            S = getattr(m, "_qk_max_logit", None)
            if S is None:
                continue
            S = S.float()
            n_heads_total += S.numel()
            max_seen = max(max_seen, float(S.max()))
            gamma = torch.clamp(self.tau / S.clamp_min(1e-12), max=1.0)   # (n_heads,)
            if bool((gamma < 1.0).any()):
                dq = int(m.d_queries)
                Wq = m.q_proj.weight                       # (n_heads*dq, d_model)
                Wk = m.k_proj.weight                       # (n_groups*dq, d_model)
                n_q = Wq.shape[0] // dq
                n_k = max(Wk.shape[0] // dq, 1)
                n_rep = max(n_q // n_k, 1)
                gq = gamma[:n_q] ** self.alpha
                # GQA: a k block serves n_rep q heads -> take the strongest clip in the group.
                g_grouped = gamma[:n_q].view(n_k, n_rep).min(dim=1).values
                gk = g_grouped ** (1.0 - self.alpha)
                Wq.mul_(gq.repeat_interleave(dq).unsqueeze(1).to(Wq.dtype))
                Wk.mul_(gk.repeat_interleave(dq).unsqueeze(1).to(Wk.dtype))
                if m.q_proj.bias is not None:
                    m.q_proj.bias.mul_(gq.repeat_interleave(dq).to(Wq.dtype))
                if m.k_proj.bias is not None:
                    m.k_proj.bias.mul_(gk.repeat_interleave(dq).to(Wk.dtype))
                n_clipped += int((gamma < 1.0).sum())
                min_gamma = min(min_gamma, float(gamma.min()))
                if self.verbose:
                    print(f"[qk-clip] step {step} {name}: {int((gamma < 1.0).sum())} heads, "
                          f"min gamma {float(gamma.min()):.4f}, max logit {float(S.max()):.2f}")
                # The observed max is now stale for this head; drop it so the next probe
                # re-measures rather than re-clipping against a pre-clip value.
                m._qk_max_logit = S * gamma
        self.disarm()
        self.was_armed = was_armed
        self.last_stats = {
            "qk_clip/max_logit": max_seen if max_seen > float("-inf") else float("nan"),
            "qk_clip/heads_clipped": float(n_clipped),
            "qk_clip/heads_total": float(n_heads_total),
            "qk_clip/min_gamma": float(min_gamma),
        }
        return self.last_stats

    @torch.no_grad()
    def observed_max(self) -> Optional[float]:
        """Largest per-head max logit seen since the last clip, or None if never probed."""
        vals = [float(m._qk_max_logit.max()) for _, m in self.modules
                if getattr(m, "_qk_max_logit", None) is not None]
        return max(vals) if vals else None
