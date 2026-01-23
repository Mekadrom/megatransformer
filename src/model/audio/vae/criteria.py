from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleMelSpectrogramLoss(nn.Module):
    """
    Multi-scale mel spectrogram loss for mel-to-mel comparison.

    Unlike MultiScaleMelLoss which operates on waveforms, this loss
    compares mel spectrograms directly at multiple resolutions by
    applying different smoothing/pooling operations.

    This is useful for VAE training where we have mel spectrograms
    directly without needing a vocoder.
    """
    def __init__(self, scales: Optional[list[int]] = None):
        super().__init__()

        # Different pooling scales for multi-resolution comparison
        if scales is None:
            scales = [1, 2, 4, 8]  # No pooling, 2x, 4x, 8x downsampling

        self.scales = scales

        # Create average pooling layers for each scale > 1
        self.pooling_layers = nn.ModuleDict()
        for scale in scales:
            if scale > 1:
                self.pooling_layers[str(scale)] = nn.AvgPool2d(
                    kernel_size=(1, scale),  # Only pool in time dimension
                    stride=(1, scale),
                )

    def forward(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-scale mel loss.

        Args:
            pred_mel: [B, 1, n_mels, T] or [B, n_mels, T] predicted mel
            target_mel: [B, 1, n_mels, T] or [B, n_mels, T] target mel
            mask: [B, T] optional mask where 1=valid, 0=padding

        Returns:
            Scalar loss value
        """
        # Ensure 4D tensors [B, 1, n_mels, T]
        if pred_mel.dim() == 3:
            pred_mel = pred_mel.unsqueeze(1)
        if target_mel.dim() == 3:
            target_mel = target_mel.unsqueeze(1)

        # Match lengths
        min_len = min(pred_mel.shape[-1], target_mel.shape[-1])
        pred_mel = pred_mel[..., :min_len]
        target_mel = target_mel[..., :min_len]

        # Prepare mask if provided
        if mask is not None:
            mask = mask[..., :min_len]  # Match length
            # Expand mask: [B, T] -> [B, 1, 1, T] for broadcasting with [B, 1, n_mels, T]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)

        total_loss = 0.0

        for scale in self.scales:
            if scale == 1:
                pred_scaled = pred_mel
                target_scaled = target_mel
                mask_scaled = mask_expanded if mask is not None else None
            else:
                pool = self.pooling_layers[str(scale)]
                pred_scaled = pool(pred_mel)
                target_scaled = pool(target_mel)
                # Downsample mask to match pooled resolution
                if mask is not None:
                    # Use max pooling for mask so any valid position in window keeps it valid
                    mask_scaled = F.max_pool2d(
                        mask_expanded.float(),
                        kernel_size=(1, scale),
                        stride=(1, scale),
                    )
                else:
                    mask_scaled = None

            diff = pred_scaled - target_scaled

            # Compute masked or unmasked L1 loss
            if mask_scaled is not None:
                abs_diff = torch.abs(diff) * mask_scaled
                valid_count = mask_scaled.sum()
                if valid_count > 0:
                    scale_loss = abs_diff.sum() / valid_count
                else:
                    scale_loss = torch.tensor(0.0, device=pred_mel.device)
            else:
                scale_loss = F.l1_loss(diff, torch.zeros_like(diff))

            total_loss = total_loss + scale_loss

        return total_loss / len(self.scales)


class AudioPerceptualLoss(nn.Module):
    """
    Combined audio perceptual loss with configurable components.

    Combines multiple audio perceptual losses with individual weights:
    - Multi-scale mel loss (mel-to-mel comparison)
    - Wav2Vec2 features (speech-specific)
    - PANNs features (general audio)

    Note: Wav2Vec2 and PANNs losses require waveforms, so a vocoder
    must be used to convert mel spectrograms to waveforms first.
    """
    def __init__(
        self,
        # Component weights (0 = disabled)
        multi_scale_mel_weight: float = 1.0,
        # Multi-scale mel settings
        mel_scales: Optional[list[int]] = None,
    ):
        super().__init__()
        self.multi_scale_mel_weight = multi_scale_mel_weight

        self.multi_scale_mel_loss = None

        if multi_scale_mel_weight > 0:
            self.multi_scale_mel_loss = MultiScaleMelSpectrogramLoss(
                scales=mel_scales,
            )

    def forward(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute audio perceptual losses.

        Args:
            pred_mel: [B, 1, n_mels, T] predicted mel spectrogram
            target_mel: [B, 1, n_mels, T] target mel spectrogram
            mask: [B, T] optional mask where 1=valid, 0=padding (for mel losses)

        Returns:
            Dict with individual losses and total perceptual loss
        """
        device = pred_mel.device
        losses = {}
        total = torch.tensor(0.0, device=device)

        # Multi-scale mel loss (works on mel spectrograms)
        if self.multi_scale_mel_loss is not None:
            ms_mel_loss = self.multi_scale_mel_loss(pred_mel, target_mel, mask=mask)
            losses["multi_scale_mel_loss"] = ms_mel_loss
            total = total + self.multi_scale_mel_weight * ms_mel_loss
        else:
            losses["multi_scale_mel_loss"] = torch.tensor(0.0, device=device)

        losses["total_perceptual_loss"] = total
        return losses
