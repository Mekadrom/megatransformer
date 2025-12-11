"""
Image discriminators for VAE-GAN training.

Implements PatchGAN-style discriminators similar to those used in
Stable Diffusion VAE and other image generation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        n_scales: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PatchDiscriminator(
                in_channels=in_channels,
                base_channels=base_channels,
                n_layers=n_layers,
                use_spectral_norm=use_spectral_norm,
            )
            for _ in range(n_scales)
        ])

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

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


class StyleGANDiscriminator(nn.Module):
    """
    StyleGAN-style discriminator with residual blocks and minibatch std.

    More powerful than PatchGAN but also more expensive.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        max_channels: int = 512,
        image_size: int = 256,
    ):
        super().__init__()

        # Calculate number of blocks based on image size
        n_blocks = 0
        size = image_size
        while size > 4:
            size //= 2
            n_blocks += 1

        # Input layer
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=1)

        # Downsampling blocks
        self.blocks = nn.ModuleList()
        channels = base_channels
        for i in range(n_blocks):
            out_channels = min(channels * 2, max_channels)
            self.blocks.append(
                DiscriminatorBlock(channels, out_channels)
            )
            channels = out_channels

        # Final layers with minibatch std
        self.final_conv = nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1)
        self.final_linear = nn.Linear(channels * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []

        x = self.input(x)
        x = F.leaky_relu(x, 0.2)
        features.append(x)

        for block in self.blocks:
            x = block(x)
            features.append(x)

        # Minibatch std
        batch_std = x.std(dim=0, keepdim=True).mean().expand(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat([x, batch_std], dim=1)

        x = self.final_conv(x)
        x = F.leaky_relu(x, 0.2)
        features.append(x)

        x = x.view(x.size(0), -1)
        x = self.final_linear(x)

        return x, features


class DiscriminatorBlock(nn.Module):
    """Residual block for StyleGAN discriminator."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(self.downsample(x))

        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.downsample(x)

        return (x + skip) / (2 ** 0.5)


# Loss functions

def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """
    Discriminator hinge loss.
    Real samples should produce positive values, fake should produce negative.
    """
    loss = 0.0
    for real, fake in zip(disc_real_outputs, disc_fake_outputs):
        loss += torch.mean(F.relu(1 - real))
        loss += torch.mean(F.relu(1 + fake))
    return loss


def generator_loss(disc_fake_outputs: list[torch.Tensor]) -> torch.Tensor:
    """
    Generator hinge loss.
    Generator wants discriminator to output positive values for fake samples.
    """
    loss = 0.0
    for fake in disc_fake_outputs:
        loss += -torch.mean(fake)
    return loss


def feature_matching_loss(
    disc_real_features: list[list[torch.Tensor]],
    disc_fake_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    Feature matching loss: L1 distance between real and fake intermediate features.
    """
    loss = 0.0
    num_layers = 0

    for real_feats, fake_feats in zip(disc_real_features, disc_fake_features):
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            loss += F.l1_loss(fake_feat, real_feat.detach())
            num_layers += 1

    return loss / num_layers if num_layers > 0 else loss


def compute_discriminator_loss(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute discriminator loss and return breakdown.

    Args:
        discriminator: The discriminator module
        real_images: Real images from the dataset
        fake_images: Fake images from the generator (detached)

    Returns:
        total_loss: Combined discriminator loss
        loss_dict: Dictionary with individual loss components
    """
    # Get discriminator outputs
    real_outputs, real_features = discriminator(real_images)
    fake_outputs, fake_features = discriminator(fake_images.detach())

    # Handle both single and multi-scale discriminators
    if not isinstance(real_outputs, list):
        real_outputs = [real_outputs]
        fake_outputs = [fake_outputs]

    d_loss = discriminator_loss(real_outputs, fake_outputs)

    loss_dict = {
        "d_loss": d_loss,
        "d_real_mean": sum(r.mean() for r in real_outputs) / len(real_outputs),
        "d_fake_mean": sum(f.mean() for f in fake_outputs) / len(fake_outputs),
    }

    return d_loss, loss_dict


def compute_generator_gan_loss(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature_matching_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute generator adversarial and feature matching losses.

    Args:
        discriminator: The discriminator module
        real_images: Real images (for feature matching)
        fake_images: Fake images from the generator (not detached)
        feature_matching_weight: Weight for feature matching loss

    Returns:
        total_loss: Combined generator GAN loss
        loss_dict: Dictionary with individual loss components
    """
    # Get discriminator outputs for fake images
    fake_outputs, fake_features = discriminator(fake_images)

    # Handle both single and multi-scale discriminators
    # Single-scale: fake_outputs is Tensor, fake_features is List[Tensor]
    # Multi-scale: fake_outputs is List[Tensor], fake_features is List[List[Tensor]]
    if not isinstance(fake_outputs, list):
        fake_outputs = [fake_outputs]
    if not isinstance(fake_features[0], list):
        fake_features = [fake_features]

    # Adversarial loss
    g_adv_loss = generator_loss(fake_outputs)

    loss_dict = {
        "g_adv_loss": g_adv_loss,
    }

    total_loss = g_adv_loss

    # Feature matching loss (optional)
    if feature_matching_weight > 0:
        with torch.no_grad():
            real_outputs, real_features = discriminator(real_images)
            # Check if this is a multi-scale discriminator (list of lists) or single (list of tensors)
            if not isinstance(real_features[0], list):
                real_features = [real_features]

        fm_loss = feature_matching_loss(real_features, fake_features)
        total_loss = total_loss + feature_matching_weight * fm_loss
        loss_dict["g_fm_loss"] = fm_loss

    return total_loss, loss_dict


# Model configs

def tiny_patch_discriminator() -> PatchDiscriminator:
    """Tiny PatchGAN discriminator (~50K params) for small VAEs."""
    return PatchDiscriminator(
        in_channels=3,
        base_channels=16,
        n_layers=2,
        use_spectral_norm=True,
    )


def mini_patch_discriminator() -> PatchDiscriminator:
    """Mini PatchGAN discriminator (~200K params)."""
    return PatchDiscriminator(
        in_channels=3,
        base_channels=32,
        n_layers=3,
        use_spectral_norm=True,
    )


def mini_multi_scale_discriminator() -> MultiScalePatchDiscriminator:
    """Mini multi-scale discriminator (~400K params)."""
    return MultiScalePatchDiscriminator(
        in_channels=3,
        base_channels=32,
        n_layers=3,
        n_scales=2,
        use_spectral_norm=True,
    )


def small_patch_discriminator() -> PatchDiscriminator:
    """Small PatchGAN discriminator (~2.8M params)."""
    return PatchDiscriminator(
        in_channels=3,
        base_channels=64,
        n_layers=3,
        use_spectral_norm=True,
    )


def tiny_multi_scale_discriminator() -> MultiScalePatchDiscriminator:
    """Tiny multi-scale discriminator (~100K params) for small VAEs."""
    return MultiScalePatchDiscriminator(
        in_channels=3,
        base_channels=16,
        n_layers=2,
        n_scales=2,
        use_spectral_norm=True,
    )


def multi_scale_discriminator() -> MultiScalePatchDiscriminator:
    """Multi-scale PatchGAN discriminator (~8.4M params)."""
    return MultiScalePatchDiscriminator(
        in_channels=3,
        base_channels=64,
        n_layers=3,
        n_scales=3,
        use_spectral_norm=True,
    )


def stylegan_discriminator(image_size: int = 256) -> StyleGANDiscriminator:
    """StyleGAN-style discriminator."""
    return StyleGANDiscriminator(
        in_channels=3,
        base_channels=64,
        max_channels=512,
        image_size=image_size,
    )


model_config_lookup = {
    "tiny_patch": tiny_patch_discriminator,
    "tiny_multi_scale": tiny_multi_scale_discriminator,
    "mini_patch": mini_patch_discriminator,
    "mini_multi_scale": mini_multi_scale_discriminator,
    "small_patch": small_patch_discriminator,
    "multi_scale": multi_scale_discriminator,
    "stylegan": stylegan_discriminator,
}