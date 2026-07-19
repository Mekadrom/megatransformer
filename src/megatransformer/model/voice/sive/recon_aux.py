"""Speaker-conditioned mel-reconstruction auxiliary head for SIVE training.

WHY THIS EXISTS
---------------
CTC supervises the character *sequence*, not per-frame acoustics, and with CTC
upsampling it happily front-loads the transcript and predicts blank over the tail
of the utterance. On CONTINUOUS features that's tolerated (the frames still carry
acoustic content for free); with a VQ bottleneck the content-less tail frames snap
to a single dominant code -> the tail collapses -> the SMG renders flatline/noise.

This head is a REGULARIZER, not a deliverable: a tiny (probe-sized) mini-SMG that
reconstructs the mel from `features + ECAPA embedding`. Because it is deliberately
small it cannot hallucinate around content-thin features, so its masked-L1 forces
EVERY real frame to carry renderable content -- structurally preventing the tail
collapse, independent of what CTC does with alignment. It is speaker-CONDITIONED
(identity comes from the ECAPA embedding via FiLM), so the features are pushed to
carry CONTENT, not speaker -- consistent with the GRL/VQ disentanglement program.

Architecture mirrors the validated `synthesis_usability.FiLMConvDecoder` probe
(the same one whose recon-L1 ranks runs in line with human listening), so there is
no new design risk -- it is that probe, co-trained instead of fitted-to-frozen.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconAuxHead(nn.Module):
    """FiLM-conditioned 1D-conv mini-SMG: features + speaker embedding -> mel.

    Keep it small (probe-sized). A powerful head would reconstruct good mel from
    content-thin features (hallucination), weakening the per-frame pressure that
    is the entire point -- so width/blocks default to the probe's proven budget.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        emb_dim: int = 192,
        n_mels: int = 80,
        width: int = 256,
        n_blocks: int = 6,
        kernel: int = 5,
    ):
        super().__init__()
        # Per-frame LayerNorm over the feature dim -> scale-robust to any
        # final_norm_type, mirrors the input-norm a real SMG applies.
        self.in_norm = nn.LayerNorm(feat_dim)
        self.stem = nn.Conv1d(feat_dim, width, 1)
        self.blocks = nn.ModuleList(
            [nn.Conv1d(width, width, kernel, padding=kernel // 2) for _ in range(n_blocks)]
        )
        # FiLM: speaker embedding -> per-block (scale, shift) over channels.
        self.film = nn.Sequential(
            nn.Linear(emb_dim, width * 2), nn.GELU(), nn.Linear(width * 2, n_blocks * 2 * width)
        )
        self.head = nn.Conv1d(width, n_mels, 1)
        self.n_blocks, self.width = n_blocks, width

    def decode(self, features: torch.Tensor, emb: torch.Tensor, t_mel: int) -> torch.Tensor:
        """features [B, T', D] (post-VQ), emb [B, emb_dim] -> mel_hat [B, n_mels, t_mel]."""
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        # Interpolate content features up to the mel frame rate (same as the probe /
        # the real SMG conditioning path), then FiLM-decode frame-parallel.
        f = features.transpose(1, 2)  # [B, D, T']
        f = F.interpolate(f.float(), size=t_mel, mode="linear", align_corners=False).to(features.dtype)
        f = self.in_norm(f.transpose(1, 2)).transpose(1, 2)  # per-frame LN over feat dim
        h = self.stem(f)
        film = self.film(emb).view(emb.size(0), self.n_blocks, 2, self.width)
        for i, conv in enumerate(self.blocks):
            g = film[:, i, 0].unsqueeze(-1)
            b = film[:, i, 1].unsqueeze(-1)
            h = h + F.gelu(g * conv(h) + b)  # FiLM speaker conditioning per block
        return self.head(h)  # [B, n_mels, t_mel]

    def forward(
        self,
        features: torch.Tensor,
        emb: torch.Tensor,
        target_mel: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (recon_mel [B, n_mels, T_mel], masked_l1 scalar).

        target_mel: [B, n_mels, T_mel]  mel_mask: [B, T_mel] True=valid (real frame).
        Loss is masked to real frames ONLY -- padded tail is zero in the target and
        must not dilute the signal (same masking the CTC/std-hinge already use).
        """
        t_mel = target_mel.shape[-1]
        recon = self.decode(features, emb, t_mel)  # [B, n_mels, T_mel]
        recon = recon.float()
        target = target_mel.float()
        if mel_mask is not None:
            m = mel_mask.unsqueeze(1).float()  # [B, 1, T_mel]
            denom = (m.sum() * recon.shape[1]).clamp(min=1.0)
            loss = ((recon - target).abs() * m).sum() / denom
        else:
            loss = (recon - target).abs().mean()
        return recon, loss
