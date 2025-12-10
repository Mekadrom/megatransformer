import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import megatransformer_utils

from model import activations


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


class VAE(nn.Module):
    def __init__(self, encoder, decoder, use_perceptual_loss=True, recon_loss_weight=1.0, perceptual_loss_weight=0.1, kl_divergence_loss_weight=0.01):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.recon_loss_weight = recon_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.kl_divergence_loss_weight = kl_divergence_loss_weight

        if use_perceptual_loss:
            self.perceptual_loss = VGGPerceptualLoss()
        else:
            self.perceptual_loss = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        recon_x = self.decoder(z)

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3])
        kl_divergence = torch.mean(kl_divergence)

        perceptual_loss = 0
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(recon_x, x)

        total_loss = self.recon_loss_weight * recon_loss + self.perceptual_loss_weight * perceptual_loss + self.kl_divergence_loss_weight * kl_divergence
        
        losses = {
            "total_loss": total_loss,
            "kl_divergence": kl_divergence,
            "recon_loss": recon_loss,
            "perceptual_loss": perceptual_loss
        }

        return recon_x, mu, logvar, losses


class VAEEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, intermediate_channels=[32, 64, 128], activation_fn: str = "silu"):
        super().__init__()

        activation = megatransformer_utils.get_activation_type(activation_fn)
        if activation not in [activations.SwiGLU, activations.Snake]:
            activation = lambda x: activation()  # drop unused arg

        channels = [in_channels] + intermediate_channels

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c)
            )
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.fc_mu = nn.Conv2d(channels[-1], latent_channels, kernel_size=3, padding=1)
        self.fc_logvar = nn.Conv2d(channels[-1], latent_channels, kernel_size=3, padding=1)

    def forward(self, x):
        for upsample in self.channel_upsample:
            x = upsample(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=3, intermediate_channels=[128, 64, 32], activation_fn: str = "silu"):
        super().__init__()

        activation = megatransformer_utils.get_activation_type(activation_fn)
        if activation not in [activations.SwiGLU, activations.Snake]:
            activation = lambda x: activation()  # drop unused arg

        channels = [latent_channels] + intermediate_channels

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c),
            )
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.final_conv = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        for upsample in self.channel_upsample:
            z = upsample(z)

        recon_x = self.final_conv(z)
        return recon_x


model_config_lookup = {
    "tiny": lambda latent_channels, **kwargs: VAE(
        encoder=VAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[32, 64],
            activation_fn="silu"
        ),
        decoder=VAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[64, 32],
            activation_fn="silu"
        ),
        **kwargs
    ),
}