import torch.nn as nn

from model import activations, get_activation_type
from model.vae import VAE


class ImageVAEEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, intermediate_channels=[32, 64, 128], activation_fn: str = "silu"):
        super().__init__()

        activation_type = get_activation_type(activation_fn)
        if activation_type not in [activations.SwiGLU, activations.Snake]:
            activation = lambda _: activation_type()  # drop unused arg
        else:
            activation = activation_type

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


class ImageVAEDecoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=3, intermediate_channels=[128, 64, 32], activation_fn: str = "silu"):
        super().__init__()

        activation_type = get_activation_type(activation_fn)
        if activation_type not in [activations.SwiGLU, activations.Snake]:
            activation = lambda _: activation_type()  # drop unused arg
        else:
            activation = activation_type

        channels = [latent_channels] + intermediate_channels

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c),
            )
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.final_conv = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, z, *_, **__):
        # speaker_embedding is ignored for image VAE (only used for audio VAE)
        for upsample in self.channel_upsample:
            z = upsample(z)

        recon_x = self.final_conv(z)
        return recon_x


model_config_lookup = {
    "tiny": lambda latent_channels, **kwargs: VAE(
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[32, 64],
            activation_fn="silu"
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[64, 32],
            activation_fn="silu"
        ),
        **kwargs
    ),
    # 200,000 total params (~100K encoder, ~100K decoder)
    "mini": lambda latent_channels, **kwargs: VAE(
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[32, 64, 128],
            activation_fn="silu"
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[128, 64, 32],
            activation_fn="silu"
        ),
        **kwargs
    ),
    # ~771K total params (~390K encoder, ~381K decoder)
    "small": lambda latent_channels, **kwargs: VAE(
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[64, 128, 256],
            activation_fn="silu"
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[256, 128, 64],
            activation_fn="silu"
        ),
        **kwargs
    ),
}
