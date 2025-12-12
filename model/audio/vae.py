import torch.nn as nn
import megatransformer_utils

from model import activations
from model.vae import VAE


class AudioVAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        latent_channels=4,
        intermediate_channels=[32, 64, 128],
        kernel_sizes=[(3, 5), (3, 5), (3, 5)],
        strides=[(2, 3), (2, 5), (2, 5)],
        activation_fn: str = "silu"
    ):
        super().__init__()

        activation_type = megatransformer_utils.get_activation_type(activation_fn)
        if activation_type not in [activations.SwiGLU, activations.Snake]:
            activation = lambda _: activation_type()  # drop unused arg
        else:
            activation = activation_type

        channels = [in_channels] + intermediate_channels

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=(kernel_size[0]//2, kernel_size[1]//2)),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c)
            )
            for in_c, out_c, kernel_size, stride in zip(channels[:-1], channels[1:], kernel_sizes, strides)
        ])

        self.fc_mu = nn.Conv2d(channels[-1], latent_channels, kernel_size=(3, 5), padding=(1, 2))
        self.fc_logvar = nn.Conv2d(channels[-1], latent_channels, kernel_size=(3, 5), padding=(1, 2))

    def forward(self, x):
        # megatransformer_utils.print_debug_tensor("AudioVAEEncoder input", x)
        for i, upsample in enumerate(self.channel_upsample):
            x = upsample(x)
            # megatransformer_utils.print_debug_tensor(f"AudioVAEEncoder after layer {i}", x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # megatransformer_utils.print_debug_tensor("AudioVAEEncoder mu", mu)
        # megatransformer_utils.print_debug_tensor("AudioVAEEncoder logvar", logvar)

        return mu, logvar


class AudioVAEDecoder(nn.Module):
    def __init__(
            self,
            latent_channels=4,
            out_channels=3,
            intermediate_channels=[128, 64, 32],
            scale_factors=[(2, 3), (2, 5), (2, 5)],
            kernel_sizes=[(3, 5), (3, 5), (3, 5)],
            activation_fn: str = "silu"
        ):
        super().__init__()

        activation_type = megatransformer_utils.get_activation_type(activation_fn)
        if activation_type not in [activations.SwiGLU, activations.Snake]:
            activation = lambda _: activation_type()  # drop unused arg
        else:
            activation = activation_type

        channels = [latent_channels] + intermediate_channels

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c),
            )
            for in_c, out_c, scale_factor, kernel_size in zip(channels[:-1], channels[1:], scale_factors, kernel_sizes)
        ])

        self.final_conv = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        # megatransformer_utils.print_debug_tensor("AudioVAEDecoder input", z)
        for i, upsample in enumerate(self.channel_upsample):
            z = upsample(z)
            # megatransformer_utils.print_debug_tensor(f"AudioVAEDecoder after layer {i}", z)

        recon_x = self.final_conv(z)
        # megatransformer_utils.print_debug_tensor("AudioVAEDecoder output", recon_x)
        return recon_x


model_config_lookup = {
    "tiny": lambda latent_channels, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[32, 64],
            kernel_sizes=[(3, 5), (3, 5)],
            strides=[(2, 3), (2, 5)],
            activation_fn="silu"
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[64, 32],
            scale_factors=[(2, 3), (2, 5)],
            kernel_sizes=[(3, 5), (3, 5)],
            activation_fn="silu"
        ),
        **kwargs
    ),
    "mini": lambda latent_channels, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[64, 128],
            kernel_sizes=[(3, 5), (3, 5)],
            strides=[(2, 3), (2, 5)],
            activation_fn="silu"
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[128, 64],
            scale_factors=[(2, 3), (2, 5)],
            kernel_sizes=[(3, 5), (3, 5)],
            activation_fn="silu"
        ),
        **kwargs
    ),
    "mini_deep": lambda latent_channels, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[32, 64, 128],
            kernel_sizes=[(3, 5), (3, 5), (3, 5)],
            strides=[(2, 3), (2, 5), (2, 5)],
            activation_fn="silu"
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[128, 64, 32],
            scale_factors=[(2, 3), (2, 5), (2, 5)],
            kernel_sizes=[(3, 5), (3, 5), (3, 5)],
            activation_fn="silu"
        ),
        **kwargs
    ),
}
