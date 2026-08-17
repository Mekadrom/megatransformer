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


def _info_nce(pred, tgt, temp):
    """Symmetric InfoNCE: pred_i must be closer to tgt_i than to tgt_j (j!=i)."""
    p = F.normalize(pred, dim=-1)
    t = F.normalize(tgt, dim=-1)
    logits = p @ t.T / temp
    labels = torch.arange(p.shape[0], device=p.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


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

    def set_whiten_stats(self, mean, std, eps: float = 1e-6):
        """Load per-dim Qwen3 target mean/std into the whitening buffers and enable it."""
        m = torch.as_tensor(mean, dtype=self.whiten_mean.dtype).flatten()
        s = torch.as_tensor(std, dtype=self.whiten_std.dtype).flatten().clamp_min(eps)
        assert m.numel() == self.whiten_mean.numel(), \
            f"whiten mean size {m.numel()} != seq_dim {self.whiten_mean.numel()}"
        self.whiten_mean.copy_(m.to(self.whiten_mean.device))
        self.whiten_std.copy_(s.to(self.whiten_std.device))
        self.whiten = True

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

        # Surface the prediction in Qwen3 space: de-whiten the head output when whitening
        # is on (identity otherwise). generate()/eval/chat render this directly.
        if self.whiten:
            mean = self.whiten_mean.view(1, 1, -1)
            std = self.whiten_std.view(1, 1, -1)
            out_seq = seq_pred * std + mean
        else:
            out_seq = seq_pred
        # Reuse the "image_clip_*" output keys so the world model / trainer / generate()
        # plumbing is shared with the SDXL adapter; pooled is None (Z-Image has none).
        out = {"image_clip_seq_pred": out_seq, "image_clip_pooled_pred": None}
        if cond_labels is not None:
            # Loss in the SAME space as seq_pred: whiten the target when enabled.
            target = ((cond_labels - self.whiten_mean.view(1, 1, -1)) / self.whiten_std.view(1, 1, -1)
                      if self.whiten else cond_labels)
            sp, sl = seq_pred, target
            # Restrict the loss to flagged rows (transcription rows carry an INPUT image,
            # not a gen target). Guard on matching length so a bad mask is ignored.
            if sample_mask is not None and sample_mask.shape[0] == seq_pred.shape[0]:
                m = sample_mask.bool()
                if int(m.sum()) == 0:
                    return out                            # no synthesis rows this batch
                sp, sl = seq_pred[m], target[m]
            sl = sl.to(sp.dtype)
            mse = F.mse_loss(sp, sl)
            loss = mse
            if self.contrastive_weight > 0 and sp.shape[0] >= 2:   # dormant at baseline
                loss = loss + self.contrastive_weight * _info_nce(
                    sp.mean(1), sl.mean(1), self.contrastive_temp)
            out["image_clip_loss"] = loss
            out["image_clip_mse_loss"] = mse.detach()
        return out
