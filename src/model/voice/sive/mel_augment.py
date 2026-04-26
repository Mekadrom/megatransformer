"""Mel-space augmentations for SIVE training.

These simulate acoustic noise and channel/mic-response variation directly in
the mel domain, avoiding the storage cost of keeping raw waveforms. Less
physically accurate than waveform-level noise injection or room impulse
response convolution, but cheap to tune and effective as a robustness prior.

All modules are no-ops outside training mode, and apply per-utterance with
a configurable probability so a fraction of each batch stays clean.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelGaussianNoise(nn.Module):
    """Additive Gaussian noise in mel space at a random target SNR.

    Per utterance: picks a random SNR uniformly in [snr_min_db, snr_max_db],
    computes per-sample signal power, derives noise power for the target SNR,
    and adds Gaussian noise of that power. Applied only to a `prob` fraction
    of the batch each step; the rest pass through unchanged.

    Note: additive Gaussian on (log-)mel is not physically equivalent to
    waveform-level acoustic noise, but regularizes SIVE features toward
    robustness to mel-space perturbations of similar magnitude.
    """

    def __init__(self, snr_min_db: float = 5.0, snr_max_db: float = 20.0, prob: float = 0.5):
        super().__init__()
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.prob = prob

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob <= 0:
            return mel_spec

        B = mel_spec.size(0)
        device = mel_spec.device

        apply_mask = torch.rand(B, device=device) < self.prob
        if not apply_mask.any():
            return mel_spec

        # Per-sample signal power, computed on the un-augmented mel.
        signal_power = (mel_spec ** 2).mean(dim=(1, 2)).clamp(min=1e-9)  # [B]

        snr_db = (
            torch.rand(B, device=device) * (self.snr_max_db - self.snr_min_db)
            + self.snr_min_db
        )
        noise_power = signal_power / (10 ** (snr_db / 10))  # [B]
        noise_std = noise_power.sqrt().view(B, 1, 1)

        noise = torch.randn_like(mel_spec) * noise_std
        # Zero out noise for samples we're not augmenting this step.
        noise = noise * apply_mask.view(B, 1, 1).to(mel_spec.dtype)
        return mel_spec + noise


class MelFrequencyResponse(nn.Module):
    """Random smooth per-band gain, simulating mic/channel EQ variation.

    Per utterance: samples a random coefficient per mel band, smooths it along
    the band dimension (low-pass across mel bands so neighbors share similar
    gain), adds 1.0, clamps positive, and multiplies the mel. Only a `prob`
    fraction of the batch is augmented each step.

    Strength controls the standard deviation of the pre-smoothing noise, so
    e.g. strength=0.3 gives roughly ±30% gain swings per band after smoothing.
    """

    def __init__(self, strength: float = 0.3, prob: float = 0.5, smoothing_kernel: int = 7):
        super().__init__()
        self.strength = strength
        self.prob = prob
        # Odd kernel gives exact same-length output with padding=(k-1)//2.
        if smoothing_kernel % 2 == 0:
            smoothing_kernel += 1
        self.smoothing_kernel = smoothing_kernel

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob <= 0 or self.strength <= 0:
            return mel_spec

        B, n_mels, _ = mel_spec.shape
        device = mel_spec.device
        dtype = mel_spec.dtype

        apply_mask = torch.rand(B, device=device) < self.prob
        if not apply_mask.any():
            return mel_spec

        raw = torch.randn(B, n_mels, device=device, dtype=dtype) * self.strength  # [B, n_mels]

        k = min(self.smoothing_kernel, n_mels if n_mels % 2 == 1 else n_mels - 1)
        if k > 1:
            padding = (k - 1) // 2
            raw = F.avg_pool1d(raw.unsqueeze(1), kernel_size=k, stride=1, padding=padding).squeeze(1)

        gain = (1.0 + raw).clamp(min=0.1).unsqueeze(-1)  # [B, n_mels, 1]

        # Where not applying, collapse gain to 1.0 so pass-through is exact.
        gain = torch.where(
            apply_mask.view(B, 1, 1),
            gain,
            torch.ones_like(gain),
        )
        return mel_spec * gain
