import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features."""
    def __init__(self, feature_layers=[3, 8, 15, 22], use_input_norm=True):
        """
        Args:
            feature_layers: indices of VGG layers to extract features from
                - 3: relu1_2 (64 channels)
                - 8: relu2_2 (128 channels)
                - 15: relu3_3 (256 channels)
                - 22: relu4_3 (512 channels)
            use_input_norm: whether to normalize input to ImageNet stats
        """
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # extract only the feature layers we need (slim)
        max_layer = max(feature_layers) + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])
        self.feature_layers = feature_layers

        # freeze
        self.features.requires_grad_(False)
        self.features.eval()

        self.use_input_norm = use_input_norm
        if use_input_norm:
            # ImageNet normalization
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()  # always eval
        return self

    def forward(self, x, target):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            target = (target - self.mean) / self.std

        loss = 0.0
        x_feat = x
        target_feat = target

        for i, layer in enumerate(self.features):
            x_feat = layer(x_feat)
            target_feat = layer(target_feat)

            if i in self.feature_layers:
                loss = loss + F.mse_loss(x_feat, target_feat)

        return loss


class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual loss wrapper.

    Requires: pip install lpips

    Model sizes:
        - 'alex' (AlexNet): ~9.1 MB, fastest, default
        - 'squeeze' (SqueezeNet): ~2.8 MB, smallest
        - 'vgg': ~58.9 MB, closest to traditional perceptual loss
    """
    def __init__(self, net: str = 'alex'):
        super().__init__()
        try:
            import lpips
        except ImportError:
            raise ImportError("LPIPS not installed. Run: pip install lpips")

        self.lpips = lpips.LPIPS(net=net, verbose=False)
        self.lpips.requires_grad_(False)
        self.lpips.eval()

    def train(self, mode=True):
        super().train(mode)
        self.lpips.eval()
        return self

    def forward(self, x, target):
        """Expects inputs in [-1, 1] range."""
        loss = self.lpips(x, target)
        return loss.mean()


class VAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        perceptual_loss_type: str = "none",  # "vgg", "lpips", "stft", or "none"
        lpips_net: str = "alex",
        recon_loss_weight: float = 1.0,
        mse_loss_weight: float = 1.0,
        l1_loss_weight: float = 0.0,
        perceptual_loss_weight: float = 0.1,
        kl_divergence_loss_weight: float = 0.01,
        free_bits: float = 0.0,  # Minimum KL per channel (0 = disabled)
        # Multi-resolution STFT loss parameters (for audio)
        stft_loss_weight: float = 0.0,  # 0 = disabled
        stft_fft_sizes: list = None,
        stft_hop_sizes: list = None,
        stft_win_lengths: list = None,
        shared_window_buffer=None,  # SharedWindowBuffer instance
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.recon_loss_weight = recon_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.kl_divergence_loss_weight = kl_divergence_loss_weight
        self.free_bits = free_bits
        self.stft_loss_weight = stft_loss_weight

        self.perceptual_loss_type = perceptual_loss_type
        if perceptual_loss_type == "vgg":
            self.perceptual_loss = VGGPerceptualLoss()
        elif perceptual_loss_type == "lpips":
            self.perceptual_loss = LPIPSLoss(net=lpips_net)
        else:
            self.perceptual_loss = None

        # Multi-resolution STFT loss for audio VAE
        self.stft_loss = None
        if stft_loss_weight > 0 and shared_window_buffer is not None:
            from model.audio.criteria import MultiResolutionSTFTLoss
            self.stft_loss = MultiResolutionSTFTLoss(
                shared_window_buffer=shared_window_buffer,
                fft_sizes=stft_fft_sizes or [256, 512, 1024, 2048],
                hop_sizes=stft_hop_sizes or [64, 128, 256, 512],
                win_lengths=stft_win_lengths or [256, 512, 1024, 2048],
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, lengths=None) -> tuple[torch.Tensor, torch.Tensor]:
        # Pass lengths to encoder if it supports it (e.g., AudioVAEEncoder)
        if lengths is not None and hasattr(self.encoder, 'time_strides'):
            return self.encoder(x, lengths=lengths)
        return self.encoder(x)

    def decode(self, z, speaker_embedding=None, lengths=None) -> torch.Tensor:
        # Pass lengths to decoder if it supports it (e.g., AudioVAEDecoder)
        if lengths is not None and hasattr(self.decoder, 'time_scale_factors'):
            return self.decoder(z, speaker_embedding=speaker_embedding, lengths=lengths)
        return self.decoder(z, speaker_embedding=speaker_embedding)

    def forward(
        self,
        x=None,
        mask=None,
        speaker_embedding=None,
        image=None,
        lengths=None,
        kl_weight_multiplier: float = 1.0,
    ):
        """
        Forward pass through VAE.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, H, T] for audio
            image: Alternative name for x (for compatibility with data collators)
            mask: Optional mask tensor [B, T] where 1 = valid, 0 = padding.
                  If provided, reconstruction loss is only computed on valid regions.
                  The mask is in the time dimension (last dim of x).
            speaker_embedding: Optional speaker embedding tensor [B, 1, D] or [B, D]
                               for conditioning the decoder (audio VAE only)
            lengths: Optional tensor [B] of valid time lengths for attention masking.
                     If provided, encoder and decoder attention will mask padded positions.
            kl_weight_multiplier: Multiplier for KL divergence weight (for KL annealing).
                                  Default 1.0 means use full kl_divergence_loss_weight.
                                  Set to 0.0 at start of training and anneal to 1.0.

        Returns:
            recon_x: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            losses: Dictionary of loss components
        """
        # Support both 'x' and 'image' as input parameter names
        if x is None and image is not None:
            x = image
        elif x is None and image is None:
            raise ValueError("Either 'x' or 'image' must be provided")

        mu, logvar = self.encode(x, lengths=lengths)
        z = self.reparameterize(mu, logvar)

        # Compute downsampled lengths for decoder attention
        # Decoder needs latent-space lengths (same as encoder output)
        latent_lengths = None
        if lengths is not None and hasattr(self.encoder, 'time_strides'):
            latent_lengths = lengths
            for stride in self.encoder.time_strides:
                latent_lengths = (latent_lengths + stride - 1) // stride

        recon_x = self.decode(z, speaker_embedding=speaker_embedding, lengths=latent_lengths)

        # Align reconstruction to input size (encoder-decoder stride may cause size mismatch)
        if recon_x.shape != x.shape:
            # Truncate or pad to match input dimensions
            slices = [slice(None)] * recon_x.dim()
            for dim in range(2, recon_x.dim()):  # Skip batch and channel dims
                if recon_x.shape[dim] > x.shape[dim]:
                    slices[dim] = slice(0, x.shape[dim])
                elif recon_x.shape[dim] < x.shape[dim]:
                    # Pad if reconstruction is smaller (shouldn't happen normally)
                    pad_size = x.shape[dim] - recon_x.shape[dim]
                    pad_dims = [0, 0] * (recon_x.dim() - dim - 1) + [0, pad_size]
                    recon_x = F.pad(recon_x, pad_dims)
            recon_x = recon_x[tuple(slices)]

        # Compute KL divergence with optional free bits
        # Per-element KL: [B, C, H, W] for images, [B, C, T] for audio
        kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if self.free_bits > 0:
            # Free bits: apply minimum KL per channel to prevent posterior collapse
            # Sum over spatial dims, mean over batch -> per-channel KL: [C]
            spatial_dims = list(range(2, mu.dim()))  # [2, 3] for 4D, [2] for 3D
            kl_per_channel = kl_per_element.sum(dim=spatial_dims).mean(dim=0)  # [C]

            # Clamp each channel's KL to at least free_bits
            kl_per_channel = torch.clamp(kl_per_channel, min=self.free_bits)

            # Sum over channels for total KL
            kl_divergence = kl_per_channel.sum()
        else:
            # Original behavior: sum over all latent dims, mean over batch
            latent_dims = list(range(1, mu.dim()))  # [1, 2, 3] for 4D, [1, 2] for 3D
            kl_divergence = kl_per_element.sum(dim=latent_dims).mean()

        # Reconstruction losses (with optional masking)
        if mask is not None:
            # Expand mask to match input shape: [B, T] -> [B, 1, 1, T] for 4D input
            # or [B, 1, T] for 3D input
            if x.dim() == 4:
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            else:
                mask_expanded = mask.unsqueeze(1)  # [B, 1, T]

            # Masked MSE loss: only compute on valid positions
            squared_error = (recon_x - x) ** 2
            masked_squared_error = squared_error * mask_expanded
            # Sum over all dims except batch, then divide by number of valid elements per sample
            valid_elements = mask_expanded.sum(dim=list(range(1, mask_expanded.dim())), keepdim=False) * x.shape[1]
            if x.dim() == 4:
                valid_elements = valid_elements * x.shape[2]  # Account for H dimension
            mse_loss = (masked_squared_error.sum(dim=list(range(1, masked_squared_error.dim()))) / (valid_elements + 1e-8)).mean()

            # Masked L1 loss
            if self.l1_loss_weight > 0:
                abs_error = torch.abs(recon_x - x)
                masked_abs_error = abs_error * mask_expanded
                l1_loss = (masked_abs_error.sum(dim=list(range(1, masked_abs_error.dim()))) / (valid_elements + 1e-8)).mean()
            else:
                l1_loss = torch.tensor(0.0, device=x.device)
        else:
            # Standard unmasked losses
            mse_loss = F.mse_loss(recon_x, x, reduction='mean')
            l1_loss = F.l1_loss(recon_x, x, reduction='mean') if self.l1_loss_weight > 0 else torch.tensor(0.0, device=x.device)

        recon_loss = self.mse_loss_weight * mse_loss + self.l1_loss_weight * l1_loss

        # Perceptual loss (not masked - operates on full images/spectrograms)
        perceptual_loss = torch.tensor(0.0, device=x.device)
        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(recon_x, x)

        # Multi-resolution STFT loss (for audio, requires waveform data passed separately)
        # This is computed externally when waveforms are available
        stft_loss = torch.tensor(0.0, device=x.device)

        # Apply KL weight multiplier for KL annealing
        effective_kl_weight = self.kl_divergence_loss_weight * kl_weight_multiplier

        total_loss = (
            self.recon_loss_weight * recon_loss
            + self.perceptual_loss_weight * perceptual_loss
            + effective_kl_weight * kl_divergence
            + self.stft_loss_weight * stft_loss
        )

        losses = {
            "total_loss": total_loss,
            "kl_divergence": kl_divergence,
            "recon_loss": recon_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "perceptual_loss": perceptual_loss,
            "stft_loss": stft_loss,
            "kl_weight_multiplier": torch.tensor(kl_weight_multiplier, device=x.device),
        }

        return recon_x, mu, logvar, losses

    def compute_stft_loss(self, pred_waveform: torch.Tensor, target_waveform: torch.Tensor) -> dict:
        """
        Compute multi-resolution STFT loss for audio.

        This should be called externally when waveforms are available (e.g., after vocoder).
        Returns dict with spectral convergence, magnitude, and complex STFT losses.
        """
        if self.stft_loss is None:
            return {
                "stft_sc_loss": torch.tensor(0.0, device=pred_waveform.device),
                "stft_mag_loss": torch.tensor(0.0, device=pred_waveform.device),
                "stft_complex_loss": torch.tensor(0.0, device=pred_waveform.device),
            }

        sc_loss, mag_loss, complex_loss = self.stft_loss(pred_waveform, target_waveform)
        return {
            "stft_sc_loss": sc_loss,
            "stft_mag_loss": mag_loss,
            "stft_complex_loss": complex_loss,
        }

    def reconstruct_with_attention(
        self,
        x: torch.Tensor,
        speaker_embedding=None,
        lengths=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, dict]:
        """
        Reconstruct input and return attention weights from encoder and decoder.

        This is a utility method for visualization/debugging during evaluation.
        It bypasses the full forward() to avoid computing losses.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, H, T] for audio
            speaker_embedding: Optional speaker embedding for conditioning
            lengths: Optional tensor [B] of valid time lengths for attention masking

        Returns:
            recon_x: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            encoder_attn_weights: Dict with 'weights' key containing [B, M, n_heads, T, T] or None
            decoder_attn_weights: Dict with 'weights' key containing [B, M, n_heads, T, T] or None
        """
        # Check if encoder supports return_attention_weights
        encoder_supports_attn = hasattr(self.encoder, 'use_attention') and self.encoder.use_attention

        # Encode with attention weights
        if encoder_supports_attn:
            enc_result = self.encoder(
                x,
                speaker_embedding=speaker_embedding,
                lengths=lengths,
                return_attention_weights=True,
            )
            mu, logvar, enc_attn_weights = enc_result
        else:
            mu, logvar = self.encode(x, lengths=lengths)
            enc_attn_weights = None

        # Sample from latent space
        z = self.reparameterize(mu, logvar)

        # Compute downsampled lengths for decoder attention
        latent_lengths = None
        if lengths is not None and hasattr(self.encoder, 'time_strides'):
            latent_lengths = lengths
            for stride in self.encoder.time_strides:
                latent_lengths = (latent_lengths + stride - 1) // stride

        # Check if decoder supports return_attention_weights
        decoder_supports_attn = hasattr(self.decoder, 'use_attention') and self.decoder.use_attention

        # Decode with attention weights
        if decoder_supports_attn:
            dec_result = self.decoder(
                z,
                speaker_embedding=speaker_embedding,
                lengths=latent_lengths,
                return_attention_weights=True,
            )
            recon_x, dec_attn_weights = dec_result
        else:
            recon_x = self.decode(z, speaker_embedding=speaker_embedding, lengths=latent_lengths)
            dec_attn_weights = None

        # Align reconstruction to input size
        if recon_x.shape != x.shape:
            slices = [slice(None)] * recon_x.dim()
            for dim in range(2, recon_x.dim()):
                if recon_x.shape[dim] > x.shape[dim]:
                    slices[dim] = slice(0, x.shape[dim])
                elif recon_x.shape[dim] < x.shape[dim]:
                    pad_size = x.shape[dim] - recon_x.shape[dim]
                    pad_dims = [0, 0] * (recon_x.dim() - dim - 1) + [0, pad_size]
                    recon_x = F.pad(recon_x, pad_dims)
            recon_x = recon_x[tuple(slices)]

        return (
            recon_x,
            mu,
            logvar,
            {"weights": enc_attn_weights},
            {"weights": dec_attn_weights},
        )
