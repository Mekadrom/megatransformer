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


class MelVTLP(nn.Module):
    """Vocal Tract Length Perturbation, applied post-hoc to log-mel.

    Piecewise-linear warp of the mel-bin axis. Per utterance: draw a warp
    factor alpha uniformly in [1 - strength, 1 + strength], compute a per-bin
    source-index lookup that's linear with slope 1/alpha below a boundary and
    linear back up to (n_mels-1, n_mels-1) above it, and resample the mel
    along the frequency axis via bilinear interpolation. Only a `prob`
    fraction of the batch is warped each step.

    This is the cheap drop-in flavor of VTLP: it operates on already-computed
    mel bins rather than perturbing the filter bank during STFT->mel. Mel
    bins are nonlinearly spaced in Hz, so the frequency-domain effect here
    is itself nonlinear — adequate as a regularizer, not a substitute for
    filter-bank-level VTLP. Pairs well with waveform-level pitch shift: VTLP
    perturbs the spectral envelope (formant locations), pitch shift moves
    the harmonic structure.
    """

    def __init__(
        self,
        strength: float = 0.1,
        prob: float = 0.5,
        boundary_frac: float = 0.7,
    ):
        super().__init__()
        self.strength = strength
        self.prob = prob
        # Fraction of the mel-bin axis to apply the linear-with-slope-1/alpha
        # warp to; above this, the warp linearly compensates so n_mels-1 maps
        # to n_mels-1 (preserves the top bin). 0.7 follows common ASR practice.
        self.boundary_frac = boundary_frac

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob <= 0 or self.strength <= 0:
            return mel_spec

        B, n_mels, T = mel_spec.shape
        if n_mels < 2 or T < 1:
            return mel_spec

        device = mel_spec.device
        dtype = mel_spec.dtype

        # Per-utterance alpha; samples not selected use alpha=1 (identity warp).
        apply_mask = torch.rand(B, device=device) < self.prob  # [B]
        alpha = 1.0 + (torch.rand(B, device=device) * 2 - 1) * self.strength  # [B]
        alpha = torch.where(apply_mask, alpha, torch.ones_like(alpha))

        # Piecewise-linear source-index lookup over output bins.
        #   below break: src = out / alpha     (slope 1/alpha, hits (0,0))
        #   above break: linear from (break, break/alpha) to (N, N)
        N = float(n_mels - 1)
        n_break = self.boundary_frac * N
        out_idx = torch.arange(n_mels, device=device, dtype=torch.float32)  # [n_mels]
        alpha_b = alpha.unsqueeze(1)  # [B, 1]

        low_src = out_idx.unsqueeze(0) / alpha_b              # [B, n_mels]
        n_break_in = n_break / alpha_b                        # [B, 1]
        slope_high = (N - n_break_in) / (N - n_break)         # [B, 1]
        high_src = n_break_in + (out_idx.unsqueeze(0) - n_break) * slope_high

        below = out_idx.unsqueeze(0) <= n_break  # [1, n_mels]
        src_idx = torch.where(below, low_src, high_src).clamp(min=0.0, max=N)  # [B, n_mels]

        # grid_sample wants input [B, C, H, W] and grid [B, H_out, W_out, 2]
        # with (x, y) in [-1, 1] where x indexes W and y indexes H. We warp
        # only the frequency (H) axis; time (W) is identity.
        y_norm = 2.0 * src_idx / max(N, 1.0) - 1.0           # [B, n_mels]
        x_norm = torch.linspace(-1.0, 1.0, T, device=device, dtype=torch.float32) if T > 1 \
            else torch.zeros(1, device=device, dtype=torch.float32)  # [T]

        y_grid = y_norm.unsqueeze(-1).expand(B, n_mels, T)   # [B, n_mels, T]
        x_grid = x_norm.view(1, 1, T).expand(B, n_mels, T)   # [B, n_mels, T]
        grid = torch.stack([x_grid, y_grid], dim=-1)          # [B, n_mels, T, 2]

        warped = F.grid_sample(
            mel_spec.float().unsqueeze(1),  # [B, 1, n_mels, T]
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(1)
        return warped.to(dtype)
