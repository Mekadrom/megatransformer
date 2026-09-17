"""Position-wise flow coda: sample one conditioning token per TRUNK position.

The AR conditioning path already has a causal backbone -- the recurrent trunk. `ARCondFlowHead`
carries its own 4-layer causal backbone only because it is handed a fixed 64-slot Q-Former
summary and has to impose an order on it itself. With the trunk in the loop that backbone is
redundant, and so is the Q-Former.

What remains is the part that actually matters: a PER-TOKEN velocity net. It maps
(noisy token, timestep, trunk state at this position) -> velocity, and nothing mixes positions.
Three consequences:

  * causality is automatic -- no mask on anything, because there is nothing to mask;
  * generation is O(L), not O(L^2): position i needs only its own trunk state, so there is no
    re-running a coda over the growing prefix;
  * length is unbounded. The number of conditioning tokens is however many trunk positions the
    image block was given, which the interleaver now sets per sample.

Dispersion is why this is a flow head and not a linear readout. A point estimate of the
conditioning is shrunk toward the conditional mean by 1-R^2 (measured alpha 0.745 vs R^2 0.738),
which the frozen decoder renders as a bland, generic scene; sampling lands on-manifold at full
dispersion per token.
"""

import torch
import torch.nn as nn

from megatransformer.model.image.ar_cond_flow_head import _FlowMLP


class PositionwiseFlowCoda(nn.Module):
    def __init__(self, seq_dim, ctx_dim, dim=512, flow_layers=3, steps=8,
                 time_sampling="logit_normal", cfg_dropout=0.0, guidance=1.0, x_skip=False):
        super().__init__()
        self.seq_dim = seq_dim
        self.steps = int(steps)
        self.time_sampling = time_sampling
        self.cfg_dropout = float(cfg_dropout)
        self.guidance = float(guidance)
        self.flow = _FlowMLP(seq_dim, ctx_dim, dim, flow_layers, x_skip=x_skip)
        self.null_ctx = nn.Parameter(torch.zeros(1, 1, ctx_dim))

    def _null(self, ctx):
        return self.null_ctx.to(ctx.dtype).expand_as(ctx)

    def _sample_tau(self, shape, device, dtype):
        if self.time_sampling == "uniform":
            return torch.rand(shape, device=device, dtype=dtype)
        return torch.sigmoid(torch.randn(shape, device=device, dtype=dtype))

    def loss(self, target, ctx, mask=None):
        """target (B,L,seq_dim) in the head's space; ctx (B,L,ctx_dim) trunk states; mask (B,L)."""
        B, L, D = target.shape
        if self.cfg_dropout > 0 and self.training:
            drop = (torch.rand(B, device=ctx.device) < self.cfg_dropout).view(-1, 1, 1)
            ctx = torch.where(drop, self._null(ctx), ctx)
        tau = self._sample_tau((B, L), target.device, torch.float32).to(target.dtype)
        noise = torch.randn_like(target)
        x_t = (1 - tau.unsqueeze(-1)) * noise + tau.unsqueeze(-1) * target
        v = self.flow(x_t.reshape(B * L, D), tau.reshape(B * L), ctx.reshape(B * L, -1))
        err = (v.reshape(B, L, D) - (target - noise)) ** 2
        if mask is None:
            return err.mean()
        # Padded positions carry no target; supervising them pulls the head toward the padding.
        m = mask.to(err.dtype).unsqueeze(-1)
        return (err * m).sum() / (m.sum() * D).clamp_min(1.0)

    @torch.no_grad()
    def sample(self, ctx, steps=None, generator=None, guidance=None):
        """ctx (B,L,ctx_dim) -> (B,L,seq_dim). Every position integrates independently."""
        steps = int(steps or self.steps)
        w = float(self.guidance if guidance is None else guidance)
        B, L, _ = ctx.shape
        dev, dt_ = ctx.device, ctx.dtype
        x = torch.randn(B, L, self.seq_dim, device=dev, dtype=dt_, generator=generator)
        use_cfg = abs(w - 1.0) > 1e-6
        c = ctx.reshape(B * L, -1)
        c_u = self._null(ctx).reshape(B * L, -1) if use_cfg else None
        dt = 1.0 / steps
        for k in range(steps):
            tau = torch.full((B * L,), k * dt, device=dev, dtype=dt_)
            xf = x.reshape(B * L, self.seq_dim)
            v = self.flow(xf, tau, c)
            if use_cfg:
                v = self.flow(xf, tau, c_u) + w * (v - self.flow(xf, tau, c_u))
            x = x + dt * v.reshape(B, L, self.seq_dim)
        return x
