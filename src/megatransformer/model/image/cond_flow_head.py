"""Conditional flow-matching head over a fixed-length conditioning sequence.

WHY (measured 2026-08-17, see [[project_dispersion_over_smoothing]]): regressing the frozen
decoder's conditioning with an MSE point loss is structurally under-dispersed -- the MSE-optimal
estimate shrinks toward the conditional mean by 1-R^2 (measured alpha 0.745 vs R^2 0.738), and the
frozen decoder renders that hedge as a bland, generic scene. A global inference gain (~1.34)
recovers ~26% of the gap, but the OPTIMAL gain varies per caption (sparse scenes ~1.2, dense ~1.8),
so a single scalar is a crude stand-in for a per-sample quantity.

This head removes the point estimate entirely: instead of predicting the conditioning, it predicts
a VELOCITY FIELD and the conditioning is produced by integrating from noise. Samples land
on-manifold at full dispersion by construction, per-caption, with no gain to tune.

Rectified-flow / linear interpolant:
    x_t = (1-t)*noise + t*target        t in [0,1]
    u   = target - noise                (constant along the path)
    loss = || v_theta(x_t, t, ctx) - u ||^2
Inference: x ~ N(0,I), then `steps` Euler steps  x <- x + dt * v_theta(x, t, ctx).

Architecture is a small DiT: self-attention over the K conditioning slots, cross-attention to the
trunk's Q-Former context, AdaLN-Zero timestep modulation. Deliberately compact (~29M at the
defaults) -- this is a per-position sampler, not a second backbone.

Modality-agnostic: it only assumes a (B, K, D) target and a (B, M, C) context, so the same head
serves image conditioning, voice feature chunks, or any other fixed-length media target.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of a (B,) float timestep in [0,1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _FlowBlock(nn.Module):
    """DiT block: AdaLN-Zero self-attn -> cross-attn to context -> MLP."""

    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # AdaLN-Zero: 3 gates + 3 (shift, scale) pairs = 9 * dim. Zero-init so each block
        # starts as the identity and the head begins as a well-behaved (if trivial) flow.
        self.ada = nn.Linear(dim, 9 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, ctx, temb):
        (sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3) = self.ada(temb).chunk(9, dim=-1)
        h = _modulate(self.n1(x), sh1, sc1)
        x = x + g1.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = _modulate(self.n2(x), sh2, sc2)
        x = x + g2.unsqueeze(1) * self.cross(h, ctx, ctx, need_weights=False)[0]
        h = _modulate(self.n3(x), sh3, sc3)
        return x + g3.unsqueeze(1) * self.mlp(h)


class CondFlowHead(nn.Module):
    """Flow-matching sampler for a (B, seq_len, seq_dim) conditioning target."""

    def __init__(self, seq_dim, ctx_dim, dim=512, n_heads=8, n_layers=4,
                 mlp_ratio=4.0, dropout=0.0, steps=8, time_sampling="logit_normal",
                 cfg_dropout=0.0, guidance=1.0, max_len=128, pos_embed=False, x_skip=False,
                 x1_pred=False, x1_sigma_min=0.02):
        super().__init__()
        self.seq_dim = seq_dim
        self.steps = int(steps)
        self.time_sampling = time_sampling
        # Classifier-free guidance. Training drops the context to a LEARNED null with
        # probability cfg_dropout, so the head learns the unconditional field alongside the
        # conditional one. Sampling then extrapolates away from the unconditional:
        #     v = v_uncond + w * (v_cond - v_uncond)
        # This is the principled version of the global `output_gain` hack: instead of scaling
        # deviation-from-the-target-mean by one constant, it pushes away from the model's OWN
        # unconditional prediction, per-sample and per-timestep. w=1 is plain conditional
        # sampling (no guidance, single forward); w>1 trades diversity for fidelity, which is
        # exactly the axis where a sampler loses to a conditional-mean point estimate.
        self.cfg_dropout = float(cfg_dropout)
        self.guidance = float(guidance)
        self.null_ctx = nn.Parameter(torch.zeros(1, 1, ctx_dim))
        self.x_in = nn.Linear(seq_dim, dim)
        self.ctx_in = nn.Linear(ctx_dim, dim)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.t_dim = dim
        self.blocks = nn.ModuleList(
            [_FlowBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, seq_dim)
        # NATIVE LENGTH + POSITIONS. Without `pos` this head is EXACTLY permutation-equivariant
        # over its output slots (verified: max|v(Px) - Pv(x)| = 6e-08) -- nothing distinguishes
        # slot i from slot j, so it emits an unordered SET and the slot order is decided by the
        # noise. That is fine for Z-Image, whose cross-attention is largely order-insensitive
        # and whose Qwen3 targets are causal (each state identifies its own position), and it is
        # how T3 reached 0.344. Once the output length VARIES, though, giving the slots an
        # identity is a real (and separately ablatable) lever, so it is a flag, not a default.
        # ── THE NOISE LEAK, AND WHY A SKIP FIXES IT ────────────────────────────────────────
        # `out` is Linear(dim -> seq_dim) with dim << seq_dim, so every velocity this head can
        # predict lies in a dim-dimensional subspace of the seq_dim-dimensional target space.
        # Euler integration only ADDS velocities to the initial noise, so the noise component
        # orthogonal to that subspace survives to the output completely untouched. MEASURED on
        # t3_2/ckpt-20000 with real contexts: 58.8% of the emitted energy at w=1, 42.6% at w=3,
        # and the projected-out signal is only 0.539 whitened std -- i.e. the head was reaching
        # plausible total dispersion by SUMMING under-dispersed content with unremovable noise,
        # not by producing dispersed content. Widening `dim` to seq_dim is the brute-force fix;
        # non-linearity is NOT a fix (the image of a dim-dimensional input is still a
        # dim-dimensional surface, and `x_in` has already discarded the noise this would need
        # to see).
        #
        # The cheap, grounded fix comes straight from the interpolant. With
        # x_t = (1-t)*noise + t*x1, the true velocity is
        #     u = x1 - noise = (x1 - x_t) / (1 - t)
        # so the full-rank part is just SUBTRACTING THE INPUT. A learned per-dim, time-conditioned
        # coefficient on x_t supplies exactly that and reaches all seq_dim directions, because the
        # product a(t) * x_t is an elementwise (diagonal) map, not a rank-dim projection.
        # ZERO-INIT so the head is bit-identical to a no-skip head at init and warm-starts from an
        # existing T3 checkpoint; unlike a zero-init output projection this still receives
        # gradient immediately (dL/da = dL/dv * x_t), so it is not subject to the dead-start trap.
        # Under CFG the skip term is IDENTICAL for the conditional and unconditional streams, so
        # it cancels in (v_c - v_u) and is applied exactly ONCE -- guidance amplifies only the
        # learned content while noise removal stays at its proper strength.
        # ── x1-PREDICTION: the FULLY grounded version of the skip ─────────────────────────
        # The learned skip above is a half-measure. Measured on the trained ws_xskip_0 head, the
        # coefficient it learns tracks the analytic -1/(1-t) early (-0.89 vs -1.00 at t=0) but
        # DIVERGES late, even going POSITIVE at t=0.875 where the exact value is -8. Compounded
        # over 8 Euler steps, 44.8% of the initial noise still survives (vs 100% with no skip).
        # Gradient descent will not discover a near-singular coefficient that is only safe when
        # applied exactly, so stop learning it: have the head predict x1 and DERIVE the velocity
        #     v = (x1_hat - x_t) / (1 - t)
        # which subtracts x_t analytically, at full rank, with the singular factor exact rather
        # than approximated. Integrated exactly this lands on x1_hat with the noise fully gone.
        # `x1_sigma_min` clamps 1-t away from 0 so the final Euler step cannot blow up.
        # NOTE this SUPERSEDES x_skip; setting both is refused in the adapter.
        self.x1_pred = bool(x1_pred)
        self.x1_sigma_min = float(x1_sigma_min)
        self.x_skip = bool(x_skip)
        if self.x_skip:
            self.skip_ada = nn.Linear(dim, seq_dim)
            nn.init.zeros_(self.skip_ada.weight)
            nn.init.zeros_(self.skip_ada.bias)

        self.pos = None
        if pos_embed:
            self.pos = nn.Parameter(torch.zeros(1, int(max_len), dim))
            nn.init.normal_(self.pos, std=0.02)
        # Zero-init the final projection + its modulation: v_theta starts at 0, so the initial
        # ODE is the identity map from noise. Standard DiT practice; avoids a violent first step.
        for m in (self.out_ada, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def velocity(self, x_t, t, ctx):
        """v_theta(x_t, t, ctx). x_t (B,K,seq_dim); t (B,) in [0,1]; ctx (B,M,ctx_dim)."""
        temb = self.t_mlp(timestep_embedding(t, self.t_dim).to(x_t.dtype))
        h = self.x_in(x_t)
        if self.pos is not None:
            if x_t.shape[1] > self.pos.shape[1]:
                raise ValueError(f"seq_len {x_t.shape[1]} exceeds max_len {self.pos.shape[1]}")
            h = h + self.pos[:, :x_t.shape[1]].to(h.dtype)
        c = self.ctx_in(ctx)
        for blk in self.blocks:
            h = blk(h, c, temb)
        shift, scale = self.out_ada(temb).chunk(2, dim=-1)
        raw = self.out(_modulate(self.out_norm(h), shift, scale))
        if self.x1_pred:
            # `raw` IS x1_hat; the -x_t term is analytic and full-rank.
            denom = (1.0 - t).clamp_min(self.x1_sigma_min).view(-1, 1, 1).to(raw.dtype)
            return (raw - x_t) / denom
        if self.x_skip:
            raw = raw + self.skip_ada(temb).unsqueeze(1) * x_t
        return raw

    def _sample_t(self, b, device, dtype):
        if self.time_sampling == "uniform":
            return torch.rand(b, device=device, dtype=dtype)
        # logit-normal (SD3): concentrates supervision on mid-trajectory timesteps, where the
        # velocity field is hardest and most of the sample quality is decided.
        return torch.sigmoid(torch.randn(b, device=device, dtype=dtype))

    def _null(self, ctx):
        return self.null_ctx.to(ctx.dtype).expand_as(ctx)

    def loss(self, target, ctx, mask=None):
        """Rectified-flow MSE. target (B,K,seq_dim) in the SAME space the head samples in.

        mask (B,K) bool marks REAL tokens when the target arrives at its native length; padded
        positions carry no target and must not be supervised or they pull the head toward
        whatever the padding happens to be.
        """
        b = target.shape[0]
        if self.cfg_dropout > 0 and self.training:
            # Per-ROW context dropout -> the head learns p(x) as well as p(x|ctx).
            drop = (torch.rand(b, device=ctx.device) < self.cfg_dropout).view(-1, 1, 1)
            ctx = torch.where(drop, self._null(ctx), ctx)
        t = self._sample_t(b, target.device, torch.float32).to(target.dtype)
        noise = torch.randn_like(target)
        tv = t.view(-1, 1, 1)
        x_t = (1 - tv) * noise + tv * target
        v = self.velocity(x_t, t, ctx)
        if mask is None:
            return F.mse_loss(v, target - noise)
        err = (v - (target - noise)) ** 2
        m = mask.to(err.dtype).unsqueeze(-1)
        return (err * m).sum() / (m.sum() * target.shape[-1]).clamp_min(1.0)

    @torch.no_grad()
    def sample(self, ctx, seq_len, steps=None, generator=None, guidance=None):
        """Integrate from noise to a sample. Euler; `steps` trades compute for fidelity.

        guidance w > 1 applies classifier-free guidance (2x cost per step, batched into a
        single forward). w=1 (default) is plain conditional sampling.
        """
        steps = int(steps or self.steps)
        w = float(self.guidance if guidance is None else guidance)
        b = ctx.shape[0]
        dev, dt_ = ctx.device, ctx.dtype
        x = torch.randn(b, seq_len, self.seq_dim, device=dev, dtype=dt_, generator=generator)
        dt = 1.0 / steps
        use_cfg = abs(w - 1.0) > 1e-6
        ctx_pair = torch.cat([ctx, self._null(ctx)], 0) if use_cfg else ctx
        for k in range(steps):
            if use_cfg:
                t = torch.full((2 * b,), k * dt, device=dev, dtype=dt_)
                v_c, v_u = self.velocity(torch.cat([x, x], 0), t, ctx_pair).chunk(2, dim=0)
                v = v_u + w * (v_c - v_u)
            else:
                t = torch.full((b,), k * dt, device=dev, dtype=dt_)
                v = self.velocity(x, t, ctx)
            x = x + dt * v
        return x
