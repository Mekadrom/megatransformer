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
    """EMA-VQ with two optional codebook-utilization levers (ViT-VQGAN, Yu et al. 2021):

    - cosine=True: L2-normalize BOTH the (projected) features and the codebook and quantize
      by cosine distance. Fixes codebook collapse / under-utilization -- the standard reason
      a K-code budget only uses ~55% of its codes.
    - code_dim < dim: quantize in a LOW-dim space (learned in_proj dim->code_dim, out_proj
      back). Low-dim codes are far easier to fill fully (under-utilization is partly curse-of-
      dimensionality). The codes are stored in code_dim; downstream (SMG / world model) still
      consume dim-D features via out_proj -- see effective_codebook() for the export.

    Both default OFF (cosine=False, code_dim=dim) -> exactly the original EMA-VQ. The EMA
    machinery is unchanged (accumulate raw projected features); cosine only changes the
    read/distance/commitment/straight-through steps (normalize at use). kmeans-init is
    incompatible with a low-dim codebook (the code space is defined by the random in_proj) --
    use data-dependent init (drop --vq_codebook_init_path) for that variant.
    """
    def __init__(self, num_codes: int, dim: int, commitment_weight: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5, dead_code_threshold: float = 1.0,
                 cosine: bool = False, code_dim: int = 0):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold
        self.cosine = cosine
        # code_dim<=0 or ==dim => no projection (identity), codes live in dim-D.
        self.code_dim = code_dim if (0 < code_dim < dim) else dim
        if self.code_dim != dim:
            self.in_proj = nn.Linear(dim, self.code_dim, bias=False)
            self.out_proj = nn.Linear(self.code_dim, dim, bias=False)
        else:
            self.in_proj = None
            self.out_proj = None

        embed = torch.randn(num_codes, self.code_dim)
        # Codebook + EMA accumulators are BUFFERS (no gradient): EMA updates them.
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("initted", torch.zeros((), dtype=torch.bool))

    def _codebook(self) -> torch.Tensor:
        """Codebook as used for distance/quantization (L2-normalized if cosine)."""
        return F.normalize(self.embed, dim=-1) if self.cosine else self.embed

    @torch.no_grad()
    def effective_codebook(self) -> torch.Tensor:
        """The dim-D codebook that downstream consumers (SMG / world model) actually see:
        the (normalized) codes projected back up through out_proj. == self.embed in the
        default (no-projection, no-cosine) case. Use THIS for the codebook export, not
        self.embed (which is code_dim and normalized-at-use)."""
        ce = self._codebook()
        return self.out_proj(ce) if self.out_proj is not None else ce

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

    @torch.no_grad()
    def load_kmeans(self, centroids: torch.Tensor):
        """Seed the codebook from a precomputed k-means codebook (utils.codebook format).

        Marks the VQ initialized so the random data-dependent init is skipped. Fit the
        k-means on the SAME features the (warm-started) encoder produces, or the centroids
        sit outside the encoder's output distribution and the seed is worse than random.
        A later checkpoint load (resume) with its own vq.embed overrides this, as intended.
        """
        if self.code_dim != self.dim:
            raise ValueError(
                "k-means codebook init is incompatible with a LOW-dim VQ (code_dim != dim): the "
                "code space is defined by the random in_proj, so a dim-D k-means codebook doesn't "
                "map into it. Use data-dependent init (drop --vq_codebook_init_path) for this variant.")
        if tuple(centroids.shape) != tuple(self.embed.shape):
            raise ValueError(f"k-means codebook shape {tuple(centroids.shape)} != VQ embed "
                             f"{tuple(self.embed.shape)} (num_codes/dim mismatch)")
        c = centroids.to(self.embed.dtype)
        self.embed.copy_(c)
        self.embed_avg.copy_(c)
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
        # Project into the code space (identity when code_dim == dim). [B,T,cd]
        z = self.in_proj(x) if self.in_proj is not None else x
        cd = z.shape[-1]
        # VQ math in fp32 to match the fp32 codebook buffers (under --bf16 autocast the
        # in_proj Linear returns bf16, which would dtype-mismatch the embed index_put in the
        # dead-code reset). .float() is differentiable, so grads still reach z. Output is cast
        # back to z's dtype below.
        flat = z.reshape(-1, cd).float()                         # [N, cd]  (raw projected features)
        if mask is None:
            valid = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
        else:
            valid = mask.reshape(-1).bool()
        flat_valid = flat[valid]

        if self.training and not bool(self.initted):
            self._init_from_data(flat_valid.detach())

        # Cosine: normalize features + codebook to the unit sphere (argmin-L2 == argmax-cosine
        # on the sphere). Default: raw. EMA still accumulates RAW projected features (below);
        # normalization is only at read/distance/commit/ST, so the EMA path is unchanged.
        fq = F.normalize(flat, dim=-1) if self.cosine else flat  # [N, cd]
        ce = self._codebook()                                    # [K, cd] (normalized if cosine)

        dist = (fq.pow(2).sum(1, keepdim=True)
                - 2 * fq @ ce.t()
                + ce.pow(2).sum(1))                              # [N, K]
        idx = dist.argmin(1)                                     # [N]
        quant_c = ce[idx]                                        # [N, cd] (the chosen code)

        if self.training and flat_valid.shape[0] > 0:
            with torch.no_grad():
                iv = idx[valid]
                onehot = F.one_hot(iv, self.num_codes).type(flat.dtype)   # [Nv, K]
                cs = onehot.sum(0)                                        # [K]
                embed_sum = onehot.t() @ flat_valid                      # [K, cd] (RAW projected)
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

        # Commitment: encoder is pulled toward its chosen (detached) code, in code space
        # (normalized if cosine). Valid frames only.
        commit = F.mse_loss(fq[valid], quant_c[valid].detach()) if flat_valid.shape[0] > 0 \
            else x.new_zeros(())
        commit = commit * self.commitment_weight

        # Straight-through in code space, then project back up to dim-D. In EVAL return the
        # code exactly (no ST roundtrip) so the exported features sit ON the codebook and
        # quantize() recovers exact ids.
        quant_c_bt = quant_c.view(B, T, cd)
        fq_bt = fq.view(B, T, cd)
        quant_st_c = fq_bt + (quant_c_bt - fq_bt).detach() if self.training else quant_c_bt
        quant_st_c = quant_st_c.to(z.dtype)  # back to the input (autocast) dtype for out_proj / downstream
        quant_st = self.out_proj(quant_st_c) if self.out_proj is not None else quant_st_c  # [B,T,D]

        with torch.no_grad():
            if valid.any():
                probs = torch.zeros(self.num_codes, device=x.device)
                probs.scatter_add_(0, idx[valid], torch.ones_like(idx[valid], dtype=probs.dtype))
                probs = probs / probs.sum().clamp_min(1)
                perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
            else:
                perplexity = x.new_zeros(())

        return quant_st, idx.view(B, T), commit, perplexity
