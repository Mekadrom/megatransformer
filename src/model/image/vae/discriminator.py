import torch
import torch.nn as nn

from config.image.vae.discriminator import MULTI_SCALE_PATCH_DISCRIMINATOR_CONFIGS, MultiScalePatchDiscriminatorConfig


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator that classifies NxN patches as real/fake.

    Uses strided convolutions to produce a spatial map of predictions,
    where each output pixel corresponds to a receptive field (patch) in the input.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        norm_f = nn.utils.spectral_norm if use_spectral_norm else lambda x: x

        # First layer: no normalization
        layers = [
            norm_f(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Intermediate layers with increasing channels
        channels = base_channels
        for i in range(1, n_layers):
            prev_channels = channels
            channels = min(base_channels * (2 ** i), 512)
            stride = 2 if i < n_layers - 1 else 1
            layers += [
                norm_f(nn.Conv2d(prev_channels, channels, kernel_size=4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Final layer: single channel output
        layers += [
            norm_f(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1)),
        ]

        self.model = nn.Sequential(*layers)

        # Store intermediate layers for feature matching
        self.layers = nn.ModuleList()
        current_layer = []
        for layer in layers:
            current_layer.append(layer)
            if isinstance(layer, nn.LeakyReLU):
                self.layers.append(nn.Sequential(*current_layer))
                current_layer = []
        if current_layer:
            self.layers.append(nn.Sequential(*current_layer))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: [B, C, H, W] image tensor

        Returns:
            output: [B, 1, H', W'] patch-wise predictions
            features: list of intermediate feature maps for feature matching
        """
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return x, features


class MultiScalePatchDiscriminator(nn.Module):
    """
    Multi-scale discriminator that operates on different image resolutions.

    Each scale uses a PatchDiscriminator on progressively downsampled images,
    capturing both fine details and global structure.
    """
    def __init__(self, config: MultiScalePatchDiscriminatorConfig):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PatchDiscriminator(
                in_channels=config.in_channels,
                base_channels=config.base_channels,
                n_layers=config.n_layers,
                use_spectral_norm=config.use_spectral_norm,
            )
            for _ in range(config.n_scales)
        ])

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "MultiScalePatchDiscriminator":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = MultiScalePatchDiscriminator.from_config("medium", n_layers=6)
        """
        if config_name not in MULTI_SCALE_PATCH_DISCRIMINATOR_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(MULTI_SCALE_PATCH_DISCRIMINATOR_CONFIGS.keys())}")

        config = MULTI_SCALE_PATCH_DISCRIMINATOR_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = MultiScalePatchDiscriminatorConfig(**config_dict)

        return cls(config)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            x: [B, C, H, W] image tensor

        Returns:
            outputs: list of discriminator outputs at each scale
            all_features: list of feature lists for feature matching
        """
        outputs = []
        all_features = []

        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            out, feats = disc(x)
            outputs.append(out)
            all_features.append(feats)

        return outputs, all_features
