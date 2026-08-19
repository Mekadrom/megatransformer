"""Autoregressive conditioning generator with a per-token flow-matching sampler (MAR-style).

Alternative to CondFlowHead's fixed-K parallel sampling: generate the conditioning sequence
ONE TOKEN AT A TIME, each token sampled by its own rectified-flow ODE conditioned on the
tokens already emitted. Factorises the same joint the parallel head models, but as
p(x_1) p(x_2|x_1) ... -- the same shape as the text and voice arms, so one mechanism serves
every modality from the shared trunk.

    causal backbone:  [BOS, x_1 .. x_{t-1}]  (+cross-attn to the trunk's Q-Former context)
                      -> z_t                          (what token t should be)
    per-token flow:   v_theta(x_t_noisy, tau, z_t)     -> velocity; integrate from noise

Precedent: MAR ("Autoregressive Image Generation without Vector Quantization") = AR backbone
+ small per-token diffusion head. The flow part is identical to CondFlowHead's rectified-flow
formulation; only the conditioning path differs.

NATIVE LENGTH. Unlike the parallel head this needs no resample to a fixed K: it emits exactly
L tokens, and L is DETERMINISTIC from the caption (tokenize it with Qwen3's tokenizer), so
there is no stop head, no EOS token, and none of the length/exposure machinery the voice arm
needed. Training is teacher-forced with a causal mask, so all L positions are supervised in
ONE forward.

INFERENCE IS SEQUENTIAL and needs a KV cache -- without one, causal self-attention over a
length-1 step is a no-op and the backbone silently ignores everything it has emitted.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatransformer.model.image.cond_flow_head import timestep_embedding, _modulate


class _ARBlock(nn.Module):
    """Causal self-attn over emitted tokens -> cross-attn to trunk context -> MLP."""

    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, ctx, attn_mask=None, cache=None):
        h = self.n1(x)
        if cache is not None:                      # inference: append this step's k/v
            k = torch.cat([cache["k"], h], 1) if cache.get("k") is not None else h
            cache["k"] = k
            a, _ = self.attn(h, k, k, need_weights=False)
        else:
            a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        h = self.n2(x)
        x = x + self.cross(h, ctx, ctx, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class _FlowMLP(nn.Module):
    """Per-token velocity net: AdaLN-modulated residual MLP conditioned on z_t and tau."""

    def __init__(self, seq_dim, cond_dim, dim, n_layers):
        super().__init__()
        self.x_in = nn.Linear(seq_dim, dim)
        self.c_in = nn.Linear(cond_dim, dim)
        self.t_dim = dim
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
                                    for _ in range(n_layers)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                                 nn.Linear(dim * 4, dim)) for _ in range(n_layers)])
        self.adas = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in range(n_layers)])
        for a in self.adas:
            nn.init.zeros_(a.weight); nn.init.zeros_(a.bias)
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, seq_dim)
        for m in (self.out_ada, self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x_t, tau, z):
        """x_t (N, seq_dim); tau (N,); z (N, cond_dim) -> velocity (N, seq_dim)."""
        c = self.c_in(z) + self.t_mlp(timestep_embedding(tau, self.t_dim).to(z.dtype))
        h = self.x_in(x_t)
        for norm, mlp, ada in zip(self.norms, self.mlps, self.adas):
            sh, sc, g = ada(c).chunk(3, dim=-1)
            h = h + g * mlp(norm(h) * (1 + sc) + sh)
        sh, sc = self.out_ada(c).chunk(2, dim=-1)
        return self.out(self.out_norm(h) * (1 + sc) + sh)


class ARCondFlowHead(nn.Module):
    def __init__(self, seq_dim, ctx_dim, dim=512, n_heads=8, n_layers=4, flow_layers=3,
                 max_len=128, dropout=0.0, steps=8, time_sampling="logit_normal",
                 cfg_dropout=0.0, guidance=1.0):
        super().__init__()
        self.seq_dim, self.dim = seq_dim, dim
        self.steps = int(steps)
        self.time_sampling = time_sampling
        self.cfg_dropout = float(cfg_dropout)
        self.guidance = float(guidance)
        self.max_len = int(max_len)

        self.tok_in = nn.Linear(seq_dim, dim)          # embeds the PREVIOUS conditioning token
        self.bos = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.ctx_in = nn.Linear(ctx_dim, dim)
        self.null_ctx = nn.Parameter(torch.zeros(1, 1, ctx_dim))
        self.blocks = nn.ModuleList([_ARBlock(dim, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(dim)
        self.flow = _FlowMLP(seq_dim, dim, dim, flow_layers)

    def _null(self, ctx):
        return self.null_ctx.to(ctx.dtype).expand_as(ctx)

    def _backbone(self, inp, ctx, caches=None, offset=0):
        """inp (B, S, dim) already embedded+positioned -> z (B, S, dim)."""
        causal = None
        if caches is None and inp.shape[1] > 1:
            S = inp.shape[1]
            causal = torch.triu(torch.full((S, S), float("-inf"), device=inp.device), diagonal=1)
        x = inp
        for i, blk in enumerate(self.blocks):
            x = blk(x, ctx, attn_mask=causal, cache=None if caches is None else caches[i])
        return self.out_norm(x)

    def _sample_tau(self, shape, device, dtype):
        if self.time_sampling == "uniform":
            return torch.rand(shape, device=device, dtype=dtype)
        return torch.sigmoid(torch.randn(shape, device=device, dtype=dtype))

    def loss(self, target, ctx, mask=None):
        """Teacher-forced rectified flow. target (B,L,seq_dim); mask (B,L) bool for real tokens."""
        B, L, D = target.shape
        if self.cfg_dropout > 0 and self.training:
            drop = (torch.rand(B, device=ctx.device) < self.cfg_dropout).view(-1, 1, 1)
            ctx = torch.where(drop, self._null(ctx), ctx)
        # teacher forcing: position t sees the TRUE tokens 1..t-1
        prev = torch.cat([self.bos.expand(B, 1, -1).to(target.dtype),
                          self.tok_in(target[:, :-1])], dim=1)
        z = self._backbone(prev + self.pos[:, :L].to(target.dtype), self.ctx_in(ctx))
        tau = self._sample_tau((B, L), target.device, torch.float32).to(target.dtype)
        noise = torch.randn_like(target)
        x_t = (1 - tau.unsqueeze(-1)) * noise + tau.unsqueeze(-1) * target
        v = self.flow(x_t.reshape(B * L, D), tau.reshape(B * L), z.reshape(B * L, self.dim))
        err = (v.reshape(B, L, D) - (target - noise)) ** 2
        if mask is None:
            return err.mean()
        # mean over REAL tokens only: padded positions carry no target and must not be
        # supervised (they would otherwise pull the head toward whatever the padding is).
        m = mask.to(err.dtype).unsqueeze(-1)
        return (err * m).sum() / (m.sum() * D).clamp_min(1.0)

    @torch.no_grad()
    def sample(self, ctx, length, steps=None, generator=None, guidance=None):
        """Emit `length` conditioning tokens autoregressively. Returns (B, length, seq_dim)."""
        steps = int(steps or self.steps)
        if length > self.max_len:
            raise ValueError(f"length {length} exceeds max_len {self.max_len} (positional table)")
        w = float(self.guidance if guidance is None else guidance)
        use_cfg = abs(w - 1.0) > 1e-6
        B, dev, dt_ = ctx.shape[0], ctx.device, ctx.dtype
        c_cond = self.ctx_in(ctx)
        c_unc = self.ctx_in(self._null(ctx)) if use_cfg else None
        # separate caches for the conditional and unconditional streams
        caches = [[{} for _ in self.blocks], [{} for _ in self.blocks]] if use_cfg else [[{} for _ in self.blocks]]
        prev = self.bos.expand(B, 1, -1).to(dt_)
        out = []
        dt = 1.0 / steps
        for i in range(length):
            inp = prev + self.pos[:, i:i + 1].to(dt_)
            z_c = self._backbone(inp, c_cond, caches=caches[0])[:, 0]
            z_u = self._backbone(inp, c_unc, caches=caches[1])[:, 0] if use_cfg else None
            x = torch.randn(B, self.seq_dim, device=dev, dtype=dt_, generator=generator)
            for k in range(steps):
                tau = torch.full((B,), k * dt, device=dev, dtype=dt_)
                v = self.flow(x, tau, z_c)
                if use_cfg:
                    v_u = self.flow(x, tau, z_u)          # ONE uncond eval, not two
                    v = v_u + w * (v - v_u)
                x = x + dt * v
            out.append(x)
            prev = self.tok_in(x).unsqueeze(1)
        return torch.stack(out, dim=1)
