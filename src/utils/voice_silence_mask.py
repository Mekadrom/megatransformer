"""Post-processing utility for masking silence in CVAE-decoded mel spectrograms.

The CVAE produces structured noise for silence regions because it was trained
on speech-only SIVE features — near-zero SIVE input is out-of-distribution.
This module detects silence in the SIVE feature space (by comparing to a
precomputed silence reference vector) and replaces the corresponding mel
frames with actual silence values.

Usage:
    mask = SilenceMask.from_checkpoint("path/to/sive/checkpoint")

    # After CVAE decode:
    mel = cvae.decode(sive_features, speaker_embeddings=spk)
    mel = mask.apply(mel, sive_features)
    # → silence regions now have mel_silence_value instead of CVAE noise
"""

import os
from typing import Optional

import torch
import torch.nn as nn


class SilenceMask(nn.Module):
    """Detects and masks silence in CVAE-decoded mel spectrograms.

    Compares each SIVE frame to a precomputed silence reference vector.
    Frames closer than `threshold` (L2 distance) are classified as silence.
    The corresponding mel frames are overwritten with `mel_silence_value`.

    The SIVE temporal stride (default 3×) means each SIVE frame maps to
    multiple mel frames. The mask is expanded accordingly.
    """

    def __init__(
        self,
        silence_vector: torch.Tensor,
        threshold: float = 5.0,
        mel_silence_value: float = -80.0,
        sive_temporal_stride: int = 3,
    ):
        """
        Args:
            silence_vector: (encoder_dim,) — the SIVE encoding of silence,
                precomputed by running silence through the SIVE encoder.
            threshold: L2 distance threshold. SIVE frames with distance to the
                silence vector below this are classified as silence. Default 5.0
                catches both zero-padded frames (distance ~1.8 from silence) and
                actual audio silence (~0), with large margin from speech (~50+).
            mel_silence_value: Value to write at silence mel frames. In log-mel
                space, -80 dB is effectively inaudible.
            sive_temporal_stride: How many mel frames each SIVE frame spans.
                For 3× downsampling SIVE, this is 3.
        """
        super().__init__()
        self.register_buffer("silence_vector", silence_vector)
        self.threshold = threshold
        self.mel_silence_value = mel_silence_value
        self.sive_temporal_stride = sive_temporal_stride

    @classmethod
    def from_checkpoint(
        cls,
        sive_checkpoint_path: str,
        threshold: float = 5.0,
        mel_silence_value: float = -80.0,
        sive_temporal_stride: int = 3,
    ) -> "SilenceMask":
        """Load from a SIVE checkpoint directory containing sive_silence_vector.pt."""
        silence_path = os.path.join(sive_checkpoint_path, "sive_silence_vector.pt")
        if not os.path.exists(silence_path):
            raise FileNotFoundError(
                f"No sive_silence_vector.pt in {sive_checkpoint_path}. "
                f"Generate it by running silence through the SIVE encoder."
            )
        data = torch.load(silence_path, map_location="cpu", weights_only=True)
        return cls(
            silence_vector=data["silence_vector"],
            threshold=threshold,
            mel_silence_value=mel_silence_value,
            sive_temporal_stride=sive_temporal_stride,
        )

    def detect_silence(self, sive_features: torch.Tensor) -> torch.Tensor:
        """Classify each SIVE frame as silence or speech.

        Args:
            sive_features: (B, C, T_sive) SIVE feature tensor.

        Returns:
            (B, T_sive) bool tensor — True where the frame is silence.
        """
        # (B, C, T) → compute L2 distance to silence_vector per frame
        # silence_vector: (C,) → (1, C, 1) for broadcasting
        ref = self.silence_vector.view(1, -1, 1)
        dist = (sive_features - ref).pow(2).sum(dim=1).sqrt()  # (B, T_sive)
        return dist < self.threshold

    def apply(
        self,
        mel: torch.Tensor,
        sive_features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replace silence regions in a CVAE-decoded mel spectrogram.

        Args:
            mel: (B, n_mels, T_mel) — CVAE-decoded mel spectrogram.
            sive_features: (B, C, T_sive) — the SIVE features that were decoded.
            lengths: Optional (B,) — real SIVE feature lengths. If provided,
                everything past lengths[b] is also marked as silence regardless
                of the SIVE content (handles padding).

        Returns:
            mel with silence regions overwritten with mel_silence_value.
        """
        silence_mask_sive = self.detect_silence(sive_features)  # (B, T_sive)

        # If lengths provided, also mark everything past the real length as silence
        if lengths is not None:
            T_sive = sive_features.shape[-1]
            length_mask = torch.arange(T_sive, device=sive_features.device).unsqueeze(0) >= lengths.view(-1, 1)
            silence_mask_sive = silence_mask_sive | length_mask

        # Expand SIVE-resolution mask to mel-resolution by repeating each frame
        # `sive_temporal_stride` times along the time axis.
        silence_mask_mel = silence_mask_sive.repeat_interleave(
            self.sive_temporal_stride, dim=1
        )  # (B, T_sive * stride)

        # Trim or pad to match mel's actual time dimension
        T_mel = mel.shape[-1]
        if silence_mask_mel.shape[-1] > T_mel:
            silence_mask_mel = silence_mask_mel[:, :T_mel]
        elif silence_mask_mel.shape[-1] < T_mel:
            # Pad with True (silence) for any remaining mel frames
            pad = torch.ones(
                mel.shape[0], T_mel - silence_mask_mel.shape[-1],
                dtype=torch.bool, device=mel.device,
            )
            silence_mask_mel = torch.cat([silence_mask_mel, pad], dim=1)

        # Expand to (B, 1, T_mel) for broadcasting over the mel bin dimension
        silence_mask_mel = silence_mask_mel.unsqueeze(1)

        # Overwrite silence regions
        mel = mel.clone()
        mel[silence_mask_mel.expand_as(mel)] = self.mel_silence_value
        return mel
