"""Differentiable Mimi semantic-content readout, for the content-cycle swap loss.

The SMG's cross-speaker (swapped) output gets NO content supervision: the recon
L1/MSE and the mel-perceptual loss need a ground-truth target mel and a conversion
has none. So on hard swaps the decoder can render the wrong words (phonemic
replacement) while still matching the target speaker. This module closes the loop
with the SAME content encoder the model is built on: vocode the swapped mel ->
24 kHz wave -> Mimi -> the continuous semantic (codebook-0) latent, which we match
against the SOURCE's own unit centroids. Content that Mimi re-reads as a different
cb0 code is penalized.

Only the SEMANTIC (WavLM-distilled) codebook-0 path is used — content, not timbre.
Everything is differentiable (no VQ argmin): we read the `input_proj` latent, which
lives in the same 256-d space as `codebook.embed`, so the target is just
`codebook.embed[source_unit_ids]` (or a cross-entropy over all centroids). Gradients
flow latent -> Mimi encoder -> Vocos -> the SMG decoder. Mimi is frozen.
"""
from typing import Optional

import torch

_MIMI_CACHE: dict = {}


def load_mimi(device, model_id: str = "kyutai/mimi"):
    """Load a frozen MimiModel (cached per device+id)."""
    key = (model_id, str(device))
    m = _MIMI_CACHE.get(key)
    if m is None:
        from transformers import MimiModel
        m = MimiModel.from_pretrained(model_id).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _MIMI_CACHE[key] = m
    return m


def mimi_centroids(mimi) -> torch.Tensor:
    """The (2048, 256) semantic codebook-0 centroid matrix (input_proj space)."""
    return mimi.quantizer.semantic_residual_vector_quantizer.layers[0].codebook.embed


def mimi_frame_rate(mimi) -> float:
    return float(mimi.config.frame_rate)  # 12.5


def mimi_semantic_latent(mimi, wav_24k: torch.Tensor) -> torch.Tensor:
    """Continuous cb0 semantic latent for a 24 kHz waveform, DIFFERENTIABLE.

    wav_24k: [B, samples] or [B, 1, samples] at 24 kHz. Returns [B, T', 256], the
    `input_proj` output (same space as the codebook centroids) — the vector Mimi
    would quantize to a cb0 code. Validated to reproduce MimiModel.encode's codes.
    """
    if wav_24k.dim() == 2:
        wav_24k = wav_24k.unsqueeze(1)  # [B, 1, samples]
    srvq = mimi.quantizer.semantic_residual_vector_quantizer
    emb = mimi.encoder(wav_24k)                                     # [B, C, t]
    emb = mimi.encoder_transformer(emb.transpose(1, 2))[0].transpose(1, 2)
    emb = mimi.downsample(emb)                                      # [B, C, T']
    lat = srvq.input_proj(emb)                                      # [B, 256, T']
    return lat.transpose(1, 2)                                      # [B, T', 256]


def mimi_content_cycle_loss(
    mimi,
    swap_wav_24k: torch.Tensor,
    source_unit_ids: torch.Tensor,
    loss_type: str = "ce",
    temperature: float = 0.1,
) -> torch.Tensor:
    """Content-preservation loss on a swapped (converted) waveform.

    swap_wav_24k: [B, (1,) samples] — the vocoded swapped output at 24 kHz.
    source_unit_ids: [B, T'] int cb0 ids of the SOURCE (pad frames == -1, ignored).
    loss_type: "ce" (cross-entropy over cb0: penalize wrong-code assignment; the
        objective most aligned with "Mimi reads the right code") or "l1" (push the
        latent toward the source centroid).
    """
    lat = mimi_semantic_latent(mimi, swap_wav_24k)                 # [B, T'', 256]
    centroids = mimi_centroids(mimi).to(lat.dtype)                 # [2048, 256]

    # Align the re-encoded frame count to the source ids (both ~ source unit count).
    T = min(lat.shape[1], source_unit_ids.shape[1])
    lat = lat[:, :T]                                               # [B, T, 256]
    ids = source_unit_ids[:, :T]                                   # [B, T]
    valid = ids >= 0                                               # pad == -1
    if valid.sum() == 0:
        return lat.sum() * 0.0

    if loss_type == "l1":
        target = centroids[ids.clamp(min=0)]                      # [B, T, 256]
        per = (lat - target).abs().mean(dim=-1)                   # [B, T]
        loss = (per * valid).sum() / valid.sum().clamp(min=1)
    else:
        # cross-entropy over the 2048 cb0 codes via COSINE-similarity logits on unit-normalized
        # vectors: logits = cos(lat, centroid)/temp, bounded in [-1/temp, 1/temp]. temp ~0.1 is
        # both SHARP (matched CE ~2, wrong-content ~9) and BOUNDED. The earlier raw -dist^2/temp
        # put logits in the hundreds (CE ~200) on 256-d vectors -> the vocoder+Mimi backward NaN'd.
        ln = torch.nn.functional.normalize(lat, dim=-1)          # [B, T, 256]
        cn = torch.nn.functional.normalize(centroids, dim=-1)    # [2048, 256]
        logits = torch.matmul(ln, cn.t()) / temperature         # [B, T, 2048]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, centroids.shape[0]),
            ids.reshape(-1).clamp(min=-1).long(),
            ignore_index=-1,
            reduction="mean",
        )

    # Finite guard: one pathological batch (e.g. a degenerate swap early in GAN) must not NaN
    # the whole run through the vocoder+Mimi backward. Skip it (zero grad) if non-finite.
    if not torch.isfinite(loss):
        return lat.sum() * 0.0
    return loss
