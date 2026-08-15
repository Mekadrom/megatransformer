"""SDXL conditioning adapter (image OUTPUT arm for the frozen-SDXL path).

Maps the recurrent trunk's image gen-query outputs to the conditioning a frozen
SDXL UNet expects, instead of predicting an image latent:

    gen-query states (B, K, d_model)
        -> self-attn mix over the K queries
        -> 77 learned output slots cross-attend the K queries
        -> seq head  : (B, 77, 2048)   # CLIP ViT-L(768) + OpenCLIP bigG(1280) concat
        -> pool head : (B, 1280)        # bigG pooled text embed (SDXL additive cond)

K (the number of gen queries) is decoupled from the fixed 77 CLIP positions by the
cross-attention, so the trunk's image-token budget is independent of the decoder
(the Q-Former-bridge idea, dimensioned for SDXL). Training target is CLIP(caption);
the tolerance probe (scripts_local/) showed frozen SDXL tolerates large *random*
conditioning error but not *semantic* error, so the loss pairs MSE with an optional
InfoNCE (discriminability) term to keep predictions in the right basin.

This module is selected by the world model ONLY when its image_coda_config is an
`SDXLAdapterConfig`; existing DiT/direct configs are untouched (backwards compatible).
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


class SDXLConditioningAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.adapter_dim
        self.contrastive_weight = float(getattr(config, "contrastive_weight", 1.0))
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
        self.pool_norm = nn.LayerNorm(d)
        self.seq_head = nn.Linear(d, config.seq_dim)
        self.pool_head = nn.Linear(d, config.pool_dim)
        # SMALL (not zero) init: keeps the initial conditioning near zero for a stable
        # regression start, but NON-zero so the InfoNCE term's F.normalize has a finite
        # gradient. Exact-zero preds make normalize singular -> exploding first backward.
        for h in (self.seq_head, self.pool_head):
            nn.init.normal_(h.weight, std=0.02); nn.init.zeros_(h.bias)

    def forward(
        self,
        encoder_hidden_states,          # (B, K, d_model) trunk image gen-query outputs
        clip_seq_labels=None,           # (B, 77, 2048) CLIP sequence target
        clip_pooled_labels=None,        # (B, 1280) CLIP pooled target
        latent_labels=None,             # accepted + ignored (shared generator call signature)
        **kw,
    ):
        x = self.in_proj(encoder_hidden_states)         # (B, K, d)
        x = self.self_enc(x)
        q = self.out_queries.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, 77, d)
        q = self.cross_dec(q, x)                        # 77 slots attend the K queries
        seq_pred = self.seq_head(self.seq_norm(q))      # (B, 77, seq_dim)
        pool_pred = self.pool_head(self.pool_norm(x.mean(dim=1)))     # (B, pool_dim)

        out = {
            "image_clip_seq_pred": seq_pred,
            "image_clip_pooled_pred": pool_pred,
        }
        if clip_seq_labels is not None and clip_pooled_labels is not None:
            mse = F.mse_loss(seq_pred, clip_seq_labels) + F.mse_loss(pool_pred, clip_pooled_labels)
            loss = mse
            if self.contrastive_weight > 0:
                contrast = (_info_nce(pool_pred, clip_pooled_labels, self.contrastive_temp)
                            + _info_nce(seq_pred.mean(1), clip_seq_labels.mean(1), self.contrastive_temp))
                loss = loss + self.contrastive_weight * contrast
            out["image_clip_loss"] = loss
            out["image_clip_mse_loss"] = mse.detach()
        return out
