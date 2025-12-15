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
        perceptual_loss_type: str = "vgg",  # "vgg", "lpips", or "none"
        lpips_net: str = "alex",
        recon_loss_weight: float = 1.0,
        mse_loss_weight: float = 1.0,
        l1_loss_weight: float = 0.0,
        perceptual_loss_weight: float = 0.1,
        kl_divergence_loss_weight: float = 0.01,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.recon_loss_weight = recon_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.kl_divergence_loss_weight = kl_divergence_loss_weight

        self.perceptual_loss_type = perceptual_loss_type
        if perceptual_loss_type == "vgg":
            self.perceptual_loss = VGGPerceptualLoss()
        elif perceptual_loss_type == "lpips":
            self.perceptual_loss = LPIPSLoss(net=lpips_net)
        else:
            self.perceptual_loss = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask=None, speaker_embedding=None):
        """
        Forward pass through VAE.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, H, T] for audio
            mask: Optional mask tensor [B, T] where 1 = valid, 0 = padding.
                  If provided, reconstruction loss is only computed on valid regions.
                  The mask is in the time dimension (last dim of x).
            speaker_embedding: Optional speaker embedding tensor [B, 1, D] or [B, D]
                               for conditioning the decoder (audio VAE only)

        Returns:
            recon_x: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            losses: Dictionary of loss components
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        recon_x = self.decoder(z, speaker_embedding=speaker_embedding)

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        kl_divergence = torch.mean(kl_divergence)

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

        total_loss = (
            self.recon_loss_weight * recon_loss
            + self.perceptual_loss_weight * perceptual_loss
            + self.kl_divergence_loss_weight * kl_divergence
        )

        losses = {
            "total_loss": total_loss,
            "kl_divergence": kl_divergence,
            "recon_loss": recon_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "perceptual_loss": perceptual_loss,
        }

        return recon_x, mu, logvar, losses
