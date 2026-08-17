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
                 mlp_ratio=4.0, dropout=0.0, steps=8, time_sampling="logit_normal"):
        super().__init__()
        self.seq_dim = seq_dim
        self.steps = int(steps)
        self.time_sampling = time_sampling
        self.x_in = nn.Linear(seq_dim, dim)
        self.ctx_in = nn.Linear(ctx_dim, dim)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.t_dim = dim
        self.blocks = nn.ModuleList(
            [_FlowBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, seq_dim)
        # Zero-init the final projection + its modulation: v_theta starts at 0, so the initial
        # ODE is the identity map from noise. Standard DiT practice; avoids a violent first step.
        for m in (self.out_ada, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def velocity(self, x_t, t, ctx):
        """v_theta(x_t, t, ctx). x_t (B,K,seq_dim); t (B,) in [0,1]; ctx (B,M,ctx_dim)."""
        temb = self.t_mlp(timestep_embedding(t, self.t_dim).to(x_t.dtype))
        h = self.x_in(x_t)
        c = self.ctx_in(ctx)
        for blk in self.blocks:
            h = blk(h, c, temb)
        shift, scale = self.out_ada(temb).chunk(2, dim=-1)
        return self.out(_modulate(self.out_norm(h), shift, scale))

    def _sample_t(self, b, device, dtype):
        if self.time_sampling == "uniform":
            return torch.rand(b, device=device, dtype=dtype)
        # logit-normal (SD3): concentrates supervision on mid-trajectory timesteps, where the
        # velocity field is hardest and most of the sample quality is decided.
        return torch.sigmoid(torch.randn(b, device=device, dtype=dtype))

    def loss(self, target, ctx):
        """Rectified-flow MSE. target (B,K,seq_dim) in the SAME space the head samples in."""
        b = target.shape[0]
        t = self._sample_t(b, target.device, torch.float32).to(target.dtype)
        noise = torch.randn_like(target)
        tv = t.view(-1, 1, 1)
        x_t = (1 - tv) * noise + tv * target
        return F.mse_loss(self.velocity(x_t, t, ctx), target - noise)

    @torch.no_grad()
    def sample(self, ctx, seq_len, steps=None, generator=None):
        """Integrate from noise to a sample. Euler; `steps` trades compute for fidelity."""
        steps = int(steps or self.steps)
        b = ctx.shape[0]
        dev, dt_ = ctx.device, ctx.dtype
        x = torch.randn(b, seq_len, self.seq_dim, device=dev, dtype=dt_, generator=generator)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((b,), k * dt, device=dev, dtype=dt_)
            x = x + dt * self.velocity(x, t, ctx)
        return x
