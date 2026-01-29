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

    must be used to convert mel spectrograms to waveforms first.
    """
    def __init__(
        self,
        vocoder,
        waveform_stft_loss,
        waveform_mel_loss,
        multi_scale_mel_loss,
        waveform_stft_weight: float = 0.0,
        waveform_mel_weight: float = 0.0,
        multi_scale_mel_weight: float = 0.0,
    ):
        super().__init__()
        self.vocoder = vocoder
        self.waveform_stft_loss = waveform_stft_loss
        self.waveform_mel_loss = waveform_mel_loss
        self.multi_scale_mel_loss = multi_scale_mel_loss
        self.waveform_stft_weight = waveform_stft_weight
        self.waveform_mel_weight = waveform_mel_weight
        self.multi_scale_mel_weight = multi_scale_mel_weight

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
        total_loss = torch.tensor(0.0, device=device)

         # Waveform-domain losses (STFT and multi-scale mel on waveforms)
        # These require vocoder and gradients flow through the frozen vocoder
        waveform_stft_loss_value = torch.tensor(0.0, device=target_mel.device)
        waveform_mel_loss_value = torch.tensor(0.0, device=target_mel.device)
        waveform_loss_enabled = (
            self.vocoder is not None
            and (self.waveform_stft_weight > 0 or self.waveform_mel_weight > 0)
        )
        if waveform_loss_enabled:
            # Run vocoder on predicted mel WITHOUT no_grad so gradients flow back
            # Vocoder weights are frozen, but gradients w.r.t. input (mel) are computed
            vocoder_outputs = self.vocoder(pred_mel.float())
            if isinstance(vocoder_outputs, dict):
                pred_waveform_grad = vocoder_outputs["pred_waveform"]
            else:
                pred_waveform_grad, _ = vocoder_outputs

            # Run vocoder on target mel WITH no_grad (don't need gradients)
            with torch.no_grad():
                target_vocoder_outputs = self.vocoder(target_mel.float())
                if isinstance(target_vocoder_outputs, dict):
                    target_waveform_grad = target_vocoder_outputs["pred_waveform"]
                else:
                    target_waveform_grad, _ = target_vocoder_outputs

            # Multi-resolution STFT loss
            if self.waveform_stft_loss is not None and self.waveform_stft_weight > 0:
                sc_loss, mag_loss, complex_stft_loss = self.waveform_stft_loss(
                    pred_waveform_grad, target_waveform_grad
                )
                waveform_stft_loss_value = sc_loss + mag_loss + complex_stft_loss
                losses["waveform_stft_loss"] = waveform_stft_loss_value
                total_loss = total_loss + self.waveform_stft_weight * waveform_stft_loss_value

            # Multi-scale mel loss (on waveforms)
            if self.waveform_mel_loss is not None and self.waveform_mel_weight > 0:
                waveform_mel_loss_value = self.waveform_mel_loss(
                    pred_waveform_grad.squeeze(1) if pred_waveform_grad.dim() == 3 else pred_waveform_grad,
                    target_waveform_grad.squeeze(1) if target_waveform_grad.dim() == 3 else target_waveform_grad
                )
                losses["waveform_mel_loss"] = waveform_mel_loss_value
                total_loss = total_loss + self.waveform_mel_weight * waveform_mel_loss_value

        # Multi-scale mel loss (works on mel spectrograms)
        if self.multi_scale_mel_loss is not None:
            ms_mel_loss = self.multi_scale_mel_loss(pred_mel, target_mel, mask=mask)
            losses["multi_scale_mel_loss"] = ms_mel_loss
            total_loss = total_loss + self.multi_scale_mel_weight * ms_mel_loss
        else:
            losses["multi_scale_mel_loss"] = torch.tensor(0.0, device=device)

        losses["total_perceptual_loss"] = total_loss
        return losses


class ArcFaceLoss(torch.nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) for learning speaker embeddings.

    Unlike simple classification which only learns a decision boundary, ArcFace
    explicitly shapes the embedding geometry by:
    - Normalizing embeddings to unit hypersphere
    - Adding angular margin to target class
    - Scaling logits to sharpen the softmax

    This forces same-speaker embeddings to cluster tightly together with angular
    separation between different speakers - the same property that makes ECAPA-TDNN
    embeddings effective for speaker verification.

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"

    Args:
        embedding_dim: Dimension of input speaker embeddings
        num_speakers: Number of speaker classes
        scale: Logit scale factor (higher = sharper softmax, typical: 30-64)
        margin: Angular margin in radians (typical: 0.2-0.5)
    """
    def __init__(
        self,
        embedding_dim: int,
        num_speakers: int,
        scale: float = 30.0,
        margin: float = 0.2,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim

        # Learnable class center weights (one per speaker)
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_speakers, embedding_dim))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute ArcFace loss.

        Args:
            embeddings: [B, D] speaker embeddings (will be L2 normalized)
            labels: [B] speaker class labels

        Returns:
            Tuple of (loss, logits, accuracy) for logging
        """
        # Handle 3D input [B, 1, D] -> [B, D]
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        # L2 normalize embeddings and weights to project onto unit hypersphere
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weights_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity (dot product of normalized vectors)
        cos_theta = F.linear(embeddings_norm, weights_norm)  # [B, num_speakers]

        # Clamp for numerical stability before acos
        cos_theta_clamped = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        # Convert to angle
        theta = torch.acos(cos_theta_clamped)

        # Add angular margin only to the target class
        # This pushes the target class embedding further from the decision boundary
        one_hot = F.one_hot(labels, num_classes=self.num_speakers).float()
        theta_m = theta + one_hot * self.margin

        # Convert back to cosine (with margin applied to target)
        cos_theta_m = torch.cos(theta_m)

        # Scale logits (higher scale = sharper probability distribution)
        logits = self.scale * cos_theta_m

        # Standard cross-entropy on scaled logits
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy for logging (use original cos_theta for predictions)
        with torch.no_grad():
            preds = cos_theta.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()

        return loss, logits, accuracy
