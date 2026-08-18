"""Z-Image conditioning adapter (image OUTPUT arm for the frozen Z-Image-Turbo path).

Maps the recurrent trunk's image gen-query outputs to the conditioning a frozen
Z-Image S3-DiT cross-attends over, instead of predicting an image latent:

    gen-query states (B, K_in, d_model)
        -> self-attn mix over the K_in queries
        -> seq_len learned output slots cross-attend the K_in queries
        -> seq head : (B, seq_len, 2560)   # Qwen3-4B penultimate hidden states

Unlike SDXL there is NO pooled vector (Z-Image has none). The conditioning length
`seq_len` is free: Z-Image cross-attends over however many tokens we supply, so the
adapter emits a fixed K and the world trainer resamples the (variable-length) Qwen3
target to the same K for a token-wise MSE (see utils/zimage_text_encoder.py).

BASELINE loss = pure MSE (config.contrastive_weight defaults to 0). The manifold /
InfoNCE / whitened-MSE / through-DiT loss tiers are deferred (see project memory);
the contrastive hook below is kept dormant so they can be switched on later.

Selected by the world model ONLY when its image_coda_config is a `ZImageAdapterConfig`;
existing DiT / direct / SDXL configs are untouched (backwards compatible).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _info_nce(pred, tgt, temp, neg=None):
    """Symmetric InfoNCE: pred_i must be closer to tgt_i than to tgt_j (j!=i).

    `neg` (N, D), if given, are EXTRA negatives (a memory queue of past targets) appended
    to the pred->tgt direction only, making that direction a (B+N)-way discrimination
    instead of B-way. The tgt->pred direction stays in-batch: there is no queue of past
    PREDICTIONS because those go stale as the model trains (the targets do not — they come
    from a frozen encoder).
    """
    p = F.normalize(pred, dim=-1)
    t = F.normalize(tgt, dim=-1)
    b = p.shape[0]
    logits = p @ t.T / temp
    labels = torch.arange(b, device=p.device)
    if neg is not None and neg.shape[0] > 0:
        logits = torch.cat([logits, p @ F.normalize(neg, dim=-1).T / temp], dim=1)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits[:, :b].T, labels))


class ZImageConditioningAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.adapter_dim
        self.contrastive_weight = float(getattr(config, "contrastive_weight", 0.0))
        self.contrastive_temp = float(getattr(config, "contrastive_temp", 0.07))

        self.in_proj = nn.Linear(config.d_model, d)
        enc = nn.TransformerEncoderLayer(
            d, config.n_heads, d * 4, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.self_enc = nn.TransformerEncoder(enc, config.n_layers)

        self.out_queries = nn.Parameter(torch.randn(config.seq_len, d) * 0.02)
        dec = nn.TransformerDecoderLayer(
            d, config.n_heads, d * 4, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.cross_dec = nn.TransformerDecoder(dec, config.n_cross_layers)

        self.seq_norm = nn.LayerNorm(d)
        self.seq_head = nn.Linear(d, config.seq_dim)
        # Small (not zero) init: near-zero conditioning for a stable regression start,
        # but non-zero so a later InfoNCE term's F.normalize has a finite gradient.
        nn.init.normal_(self.seq_head.weight, std=0.02)
        nn.init.zeros_(self.seq_head.bias)

        # Tier-1 InfoNCE projection head (dormant unless contrastive_weight>0). Contrast the
        # mean-pooled conditioning in a LEARNED space — raw cosine on LM features is
        # near-saturated (0.999+ even for wrong captions), so it can't separate them.
        pdim = int(getattr(config, "contrastive_proj_dim", 256))
        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.seq_dim, pdim), nn.GELU(), nn.Linear(pdim, pdim))

        # MoCo-style memory queue of past TARGET conditioning (pooled, in loss space).
        # In-batch InfoNCE is only a batch_size-way task -- at batch 8 that is solved to
        # ~0 loss almost immediately and contributes no gradient. The queue raises it to a
        # (B + queue_size)-way task, which is where contrastive objectives actually bite.
        #
        # Unlike MoCo there is NO momentum encoder and the stored vectors are NOT
        # pre-projected: the targets come from a FROZEN Qwen3, so raw target features never
        # go stale, and re-projecting the whole queue through the CURRENT head each step
        # gives exactly-consistent (zero-lag) negatives. Gradient is allowed to flow through
        # the queue's projection -- that is what teaches the head to spread targets apart,
        # and it cannot degenerate because a queued vector is drawn from the same
        # distribution as the positives (it WAS a positive on an earlier step).
        #
        # Non-persistent: kept out of the checkpoint (42MB at 4096x2560) since it is a
        # transient training artifact and refills in queue_size/batch_size steps.
        self.contrastive_queue_size = int(getattr(config, "contrastive_queue_size", 0))
        if self.contrastive_queue_size > 0:
            self.register_buffer(
                "contrastive_queue",
                torch.zeros(self.contrastive_queue_size, config.seq_dim),
                persistent=False)
            self.register_buffer("contrastive_queue_ptr", torch.zeros((), dtype=torch.long),
                                 persistent=False)
            self.register_buffer("contrastive_queue_fill", torch.zeros((), dtype=torch.long),
                                 persistent=False)

        # Tier-0 whitening: regress in a per-dim z-scored Qwen3 space. Centering removes
        # the massive near-constant outlier dims (LLM "massive activations"), scaling
        # equalizes each dim's loss contribution -> attacks the MSE-mean mode-collapse.
        # Stats (mean/std over the target distribution) are injected by the trainer via
        # set_whiten_stats and persist as buffers (so eval/chat de-whiten from the ckpt).
        # Identity (mean 0, std 1) by default = no-op. The head outputs WHITENED space when
        # on; the surfaced image_clip_seq_pred is de-whitened back to Qwen3 space.
        self.whiten = bool(getattr(config, "whiten_target", False))
        self.register_buffer("whiten_mean", torch.zeros(config.seq_dim))
        self.register_buffer("whiten_std", torch.ones(config.seq_dim))

        # Inference-only dispersion correction (see ZImageAdapterConfig.output_gain): scales
        # the WHITENED prediction before de-whitening, undoing the shrinkage an MSE-optimal
        # point estimate is forced into. Touches only the surfaced prediction, never the loss.
        self.output_gain = float(getattr(config, "output_gain", 1.0))

        # TIER-3: replace the point estimate with a flow-matching SAMPLER over the whitened
        # conditioning. An MSE point head is structurally under-dispersed (shrinks by 1-R^2)
        # and the frozen DiT renders that as bland; a sampler lands on-manifold at full
        # dispersion per-caption, with no global gain to tune. The Q-Former output is the
        # conditioning context; seq_head is kept as a cheap monitoring/aux point estimate
        # (flow_aux_mse_weight) so alpha/R^2 stay measurable against the regression runs.
        self.flow_head = None
        if bool(getattr(config, "flow_head", False)):
            from megatransformer.model.image.cond_flow_head import CondFlowHead
            self.flow_head = CondFlowHead(
                seq_dim=config.seq_dim, ctx_dim=d,
                dim=int(getattr(config, "flow_dim", 512)),
                n_heads=int(getattr(config, "flow_heads", 8)),
                n_layers=int(getattr(config, "flow_layers", 4)),
                dropout=config.dropout,
                steps=int(getattr(config, "flow_steps", 8)),
                time_sampling=getattr(config, "flow_time_sampling", "logit_normal"))
        self.flow_aux_mse_weight = float(getattr(config, "flow_aux_mse_weight", 0.1))

    def set_whiten_stats(self, mean, std, eps: float = 1e-6):
        """Load per-dim Qwen3 target mean/std into the whitening buffers and enable it."""
        m = torch.as_tensor(mean, dtype=self.whiten_mean.dtype).flatten()
        s = torch.as_tensor(std, dtype=self.whiten_std.dtype).flatten().clamp_min(eps)
        assert m.numel() == self.whiten_mean.numel(), \
            f"whiten mean size {m.numel()} != seq_dim {self.whiten_mean.numel()}"
        self.whiten_mean.copy_(m.to(self.whiten_mean.device))
        self.whiten_std.copy_(s.to(self.whiten_std.device))
        self.whiten = True

    def _queue_negatives(self):
        """Filled slice of the target memory queue, or None if disabled/empty.

        CLONED, not a view: _enqueue writes into the buffer in place during the same
        forward, which would otherwise bump the version of the exact tensor the projection
        head consumed and make backward fail ("modified by an inplace operation").
        """
        if self.contrastive_queue_size <= 0:
            return None
        fill = int(self.contrastive_queue_fill)
        return self.contrastive_queue[:fill].clone() if fill > 0 else None

    @torch.no_grad()
    def _enqueue(self, vecs):
        """Ring-buffer write of pooled target vectors (B, seq_dim) into the memory queue."""
        if self.contrastive_queue_size <= 0:
            return
        q = self.contrastive_queue
        v = vecs.detach().to(q.dtype).to(q.device)
        n, cap = v.shape[0], q.shape[0]
        if n >= cap:                                   # batch alone overfills: keep the tail
            q.copy_(v[-cap:])
            self.contrastive_queue_ptr.zero_()
            self.contrastive_queue_fill.fill_(cap)
            return
        ptr = int(self.contrastive_queue_ptr)
        end = ptr + n
        if end <= cap:
            q[ptr:end] = v
        else:                                          # wrap
            head = cap - ptr
            q[ptr:] = v[:head]
            q[:end - cap] = v[head:]
        self.contrastive_queue_ptr.fill_(end % cap)
        self.contrastive_queue_fill.fill_(min(cap, int(self.contrastive_queue_fill) + n))

    def forward(
        self,
        encoder_hidden_states,          # (B, K_in, d_model) trunk image gen-query outputs
        cond_labels=None,               # (B, seq_len, 2560) resampled Qwen3 target
        sample_mask=None,               # (B,) bool: rows to include in the loss (synthesis only)
        latent_labels=None,             # accepted + ignored (shared generator call signature)
        **kw,
    ):
        x = self.in_proj(encoder_hidden_states)          # (B, K_in, d)
        x = self.self_enc(x)
        q = self.out_queries.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, seq_len, d)
        q = self.cross_dec(q, x)                         # seq_len slots attend the K_in queries
        seq_pred = self.seq_head(self.seq_norm(q))       # (B, seq_len, seq_dim); WHITENED space if self.whiten

        # WHAT GETS SURFACED (in whitened space):
        #   flow head + no labels (inference) -> INTEGRATE FROM NOISE. Full-dispersion,
        #     on-manifold, per-caption -- the whole point of T3.
        #   otherwise -> the point head (also used during training, where surfacing the
        #     cheap regression keeps the alpha/R^2 diagnostics comparable to the T0/T1 runs
        #     and avoids paying `steps` extra head passes on every training forward).
        if self.flow_head is not None and cond_labels is None:
            # Seeded sampling matters for EVAL COMPARABILITY: the head emits a DISTRIBUTION, so
            # an unseeded draw makes checkpoint-to-checkpoint differences a mix of training
            # progress and sampling noise. Setting `.flow_generator` pins the noise so the same
            # draw is compared across checkpoints; leave it None for genuinely random samples.
            seq_surf = self.flow_head.sample(
                q, seq_pred.shape[1],
                generator=kw.get("flow_generator", getattr(self, "flow_generator", None)))
        else:
            seq_surf = seq_pred

        # Surface the prediction in Qwen3 space: de-whiten the head output when whitening
        # is on (identity otherwise). generate()/eval/chat render this directly.
        if self.whiten:
            mean = self.whiten_mean.view(1, 1, -1)
            std = self.whiten_std.view(1, 1, -1)
            # output_gain scales in WHITENED space (target mean 0), i.e. it scales the
            # deviation from the target mean and leaves the mean itself alone.
            out_seq = (seq_surf * self.output_gain) * std + mean
        else:
            if self.output_gain != 1.0:
                raise ValueError(
                    "output_gain requires whiten_target: without whitening the prediction is in "
                    "raw Qwen3 space, where scaling would also blow up the massive near-constant "
                    "dims (|mean| up to 1013) instead of scaling the deviation from the mean.")
            out_seq = seq_surf
        # Reuse the "image_clip_*" output keys so the world model / trainer / generate()
        # plumbing is shared with the SDXL adapter; pooled is None (Z-Image has none).
        out = {"image_clip_seq_pred": out_seq, "image_clip_pooled_pred": None}
        if cond_labels is not None:
            # Loss in the SAME space as seq_pred: whiten the target when enabled.
            target = ((cond_labels - self.whiten_mean.view(1, 1, -1)) / self.whiten_std.view(1, 1, -1)
                      if self.whiten else cond_labels)
            sp, sl, ctx = seq_pred, target, q
            # Restrict the loss to flagged rows (transcription rows carry an INPUT image,
            # not a gen target). Guard on matching length so a bad mask is ignored.
            if sample_mask is not None and sample_mask.shape[0] == seq_pred.shape[0]:
                m = sample_mask.bool()
                if int(m.sum()) == 0:
                    return out                            # no synthesis rows this batch
                sp, sl, ctx = seq_pred[m], target[m], q[m]
            sl = sl.to(sp.dtype)
            mse = F.mse_loss(sp, sl)
            if self.flow_head is not None:
                # T3: the flow-matching objective IS the training signal. The point-head MSE
                # is kept only as a small auxiliary so seq_pred stays a usable diagnostic
                # (alpha / R^2 / retrieval) against the regression runs -- set the weight to
                # 0 to train a pure sampler.
                flow = self.flow_head.loss(sl, ctx)
                loss = flow + self.flow_aux_mse_weight * mse
                out["image_flow_loss"] = flow.detach()
            else:
                loss = mse
            if self.contrastive_weight > 0 and sp.shape[0] >= 2:   # Tier-1: needs >=2 rows
                tgt_pool = sl.mean(1)                              # (B, seq_dim), loss space
                neg = self._queue_negatives()
                nce = _info_nce(
                    self.contrastive_proj(sp.mean(1)),
                    self.contrastive_proj(tgt_pool),
                    self.contrastive_temp,
                    neg=self.contrastive_proj(neg.to(sp.dtype)) if neg is not None else None)
                loss = loss + self.contrastive_weight * nce
                out["image_contrastive_loss"] = nce.detach()
                out["image_contrastive_negatives"] = torch.as_tensor(
                    float(0 if neg is None else neg.shape[0]), device=sp.device)
                # Enqueue AFTER the loss so the current batch's own targets are never in
                # its own negative set (they are already the in-batch negatives).
                if self.training:
                    self._enqueue(tgt_pool)
            out["image_clip_loss"] = loss
            out["image_clip_mse_loss"] = mse.detach()
        return out
