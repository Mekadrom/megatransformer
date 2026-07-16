"""EMA vector-quantization bottleneck for SIVE.

Placed on the POST-final-norm encoder output (what the CTC and GRL heads read), so the
codebook is what both objectives supervise directly: CTC forces the codes to carry
phonetic content, the GRL adversary forces them to drop speaker, and the K-code budget is
the capacity constraint that makes the disentanglement actually bite (a continuous encoder
has spare capacity to smuggle speaker past the GRL; K discrete codes force it to choose,
and content wins because it is the primary objective).

EMA codebook (van den Oord et al. 2017), not a gradient-trained one: the codes track the
encoder output by exponential moving average, which is markedly more stable than the naive
codebook loss. Collapse mitigations included, since a collapsed codebook (few codes used)
is the standard VQ failure: data-dependent init from the first batch, Laplace-smoothed
cluster sizes, and dead-code reset (unused codes re-seeded from live encoder outputs).

Straight-through estimator passes the downstream (CTC/GRL) gradients through the argmin to
the encoder; the commitment loss is the encoder's own pull toward its chosen code. Only the
returned commitment loss goes into the training objective -- the codebook itself is updated
by EMA, not by gradient.

Padding matters: only VALID frames update the codebook and enter the commitment loss.
Feeding padded frames would drag codes toward whatever fills the pad region.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes: int, dim: int, commitment_weight: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5, dead_code_threshold: float = 1.0):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold

        embed = torch.randn(num_codes, dim)
        # Codebook + EMA accumulators are BUFFERS (no gradient): EMA updates them.
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("initted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _init_from_data(self, flat_valid: torch.Tensor):
        """Data-dependent init: seed the codebook from real encoder outputs so codes start
        inside the feature distribution (random-Gaussian init is a classic early-collapse
        cause). Sample-with-replacement if the first batch has fewer valid frames than codes."""
        n = flat_valid.shape[0]
        if n == 0:
            return
        idx = torch.randint(0, n, (self.num_codes,), device=flat_valid.device)
        chosen = flat_valid[idx]
        self.embed.copy_(chosen)
        self.embed_avg.copy_(chosen)
        self.cluster_size.fill_(1.0)
        self.initted.fill_(True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """x: [B, T, D] post-norm features. mask: [B, T] True=valid (None => all valid).

        Returns (quantized [B, T, D] with straight-through grad, code indices [B, T] long,
        commitment_loss scalar, perplexity scalar). Padded positions in the returned tensor
        are the quantized value too, but they are excluded from the loss and EMA and their
        indices are meaningless -- consumers mask by feature_lengths as before.
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)                                   # [N, D]
        if mask is None:
            valid = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
        else:
            valid = mask.reshape(-1).bool()
        flat_valid = flat[valid]

        if self.training and not bool(self.initted):
            self._init_from_data(flat_valid.detach())

        # Nearest code by L2 (expanded form avoids a full cdist allocation).
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embed.t()
                + self.embed.pow(2).sum(1))                       # [N, K]
        idx = dist.argmin(1)                                       # [N]
        quant = self.embed[idx].view(B, T, D)

        if self.training and flat_valid.shape[0] > 0:
            with torch.no_grad():
                iv = idx[valid]
                onehot = F.one_hot(iv, self.num_codes).type(flat.dtype)   # [Nv, K]
                cs = onehot.sum(0)                                        # [K]
                embed_sum = onehot.t() @ flat_valid                      # [K, D]
                self.cluster_size.mul_(self.decay).add_(cs, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                # Laplace smoothing so a briefly-unused code doesn't divide by ~0.
                n = self.cluster_size.sum()
                smoothed = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
                self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
                # Dead-code reset: re-seed codes that fell below threshold usage from live
                # frames, so a collapsing codebook re-expands instead of shrinking further.
                dead = self.cluster_size < self.dead_code_threshold
                if dead.any():
                    nd = int(dead.sum())
                    ridx = torch.randint(0, flat_valid.shape[0], (nd,), device=flat_valid.device)
                    seed = flat_valid[ridx]
                    self.embed[dead] = seed
                    self.embed_avg[dead] = seed
                    self.cluster_size[dead] = 1.0

        # Commitment: encoder is pulled toward its chosen (detached) code. Valid frames only.
        commit = F.mse_loss(flat_valid, quant.reshape(-1, D)[valid].detach()) if flat_valid.shape[0] > 0 \
            else x.new_zeros(())
        commit = commit * self.commitment_weight

        # Straight-through in TRAIN so downstream (CTC/GRL) grads reach the encoder. In EVAL
        # return the code exactly: no grad is needed, and x + (quant - x) roundtrips x, which
        # loses precision (bf16/float rounding) and would leave the "quantized" features a hair
        # off the codebook -- the downstream dataset needs them ON the codebook so quantize()
        # recovers exact ids. So eval returns bit-exact codebook rows.
        quant_st = x + (quant - x).detach() if self.training else quant

        with torch.no_grad():
            if valid.any():
                probs = torch.zeros(self.num_codes, device=x.device)
                probs.scatter_add_(0, idx[valid], torch.ones_like(idx[valid], dtype=probs.dtype))
                probs = probs / probs.sum().clamp_min(1)
                perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
            else:
                perplexity = x.new_zeros(())

        return quant_st, idx.view(B, T), commit, perplexity
