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


def add_instance_noise(
    images: torch.Tensor,
    std: float = 0.1,
) -> torch.Tensor:
    """
    Add instance noise to images for discriminator regularization.

    This prevents the discriminator from finding shortcuts based on artifacts
    (blur, color shifts) and forces it to learn actual image structure.

    Args:
        images: [B, C, H, W] image tensor
        std: Standard deviation of Gaussian noise (decay over training)

    Returns:
        Noisy images
    """
    if std <= 0:
        return images
    return images + std * torch.randn_like(images)


def r1_gradient_penalty(
    real_images: torch.Tensor,
    discriminator: nn.Module,
) -> torch.Tensor:
    """
    R1 gradient penalty (Mescheder et al., 2018).

    Penalizes the gradient norm on real images, encouraging the discriminator
    to have flat responses around real data. This stabilizes training and
    prevents the discriminator from focusing only on detecting fake artifacts.

    Args:
        real_images: [B, C, H, W] real image tensor (will enable gradients)
        discriminator: The discriminator module

    Returns:
        R1 penalty (scalar tensor)
    """
    real_images = real_images.detach().requires_grad_(True)

    real_outputs, _ = discriminator(real_images)

    # Handle multi-scale discriminators
    if isinstance(real_outputs, list):
        real_outputs = sum(r.mean() for r in real_outputs)
    else:
        real_outputs = real_outputs.sum()

    # Compute gradients
    grads = torch.autograd.grad(
        outputs=real_outputs,
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # R1 = E[||∇D(x)||²]
    penalty = grads.pow(2).reshape(grads.size(0), -1).sum(1).mean()

    return penalty


def r2_gradient_penalty(
    fake_images: torch.Tensor,
    discriminator: nn.Module,
) -> torch.Tensor:
    """
    R2 gradient penalty - same as R1 but on fake images.

    Less commonly used than R1, but can help if generator is unstable.

    Args:
        fake_images: [B, C, H, W] fake image tensor (will enable gradients)
        discriminator: The discriminator module

    Returns:
        R2 penalty (scalar tensor)
    """
    fake_images = fake_images.detach().requires_grad_(True)

    fake_outputs, _ = discriminator(fake_images)

    if isinstance(fake_outputs, list):
        fake_outputs = sum(f.sum() for f in fake_outputs)
    else:
        fake_outputs = fake_outputs.sum()

    grads = torch.autograd.grad(
        outputs=fake_outputs,
        inputs=fake_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    penalty = grads.pow(2).reshape(grads.size(0), -1).sum(1).mean()

    return penalty


class InstanceNoiseScheduler:
    """
    Scheduler for decaying instance noise over training.

    Starts with high noise to prevent shortcut learning, then decays
    to let the discriminator learn finer details.

    Usage:
        noise_scheduler = InstanceNoiseScheduler(initial_std=0.2, decay_steps=50000)
        for step in range(num_steps):
            noise_std = noise_scheduler.get_std(step)
            real_noisy = add_instance_noise(real, noise_std)
            fake_noisy = add_instance_noise(fake, noise_std)
    """
    def __init__(
        self,
        initial_std: float = 0.2,
        final_std: float = 0.0,
        decay_steps: int = 50000,
        decay_type: str = "linear",  # "linear", "cosine", "exponential"
    ):
        self.initial_std = initial_std
        self.final_std = final_std
        self.decay_steps = decay_steps
        self.decay_type = decay_type

    def get_std(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.final_std

        progress = step / self.decay_steps

        if self.decay_type == "linear":
            return self.initial_std + (self.final_std - self.initial_std) * progress
        elif self.decay_type == "cosine":
            import math
            return self.final_std + 0.5 * (self.initial_std - self.final_std) * (1 + math.cos(math.pi * progress))
        elif self.decay_type == "exponential":
            import math
            # Exponential decay: std = initial * exp(-5 * progress)
            # Factor of 5 means ~1% of initial at end
            return self.final_std + (self.initial_std - self.final_std) * math.exp(-5 * progress)
        else:
            return self.initial_std


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
    return loss / (2 * len(disc_real_outputs))


def generator_loss(disc_fake_outputs: list[torch.Tensor]) -> torch.Tensor:
    """
    Generator hinge loss.
    Generator wants discriminator to output positive values for fake samples.
    """
    loss = 0.0
    for fake in disc_fake_outputs:
        loss += -torch.mean(fake)
    return loss / len(disc_fake_outputs)


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
    real_outputs, _real_features = discriminator(real_images)
    fake_outputs, _fake_features = discriminator(fake_images.detach())

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
            _real_outputs, real_features = discriminator(real_images)
            # Check if this is a multi-scale discriminator (list of lists) or single (list of tensors)
            if not isinstance(real_features[0], list):
                real_features = [real_features]

        fm_loss = feature_matching_loss(real_features, fake_features)
        total_loss = total_loss + feature_matching_weight * fm_loss
        loss_dict["g_fm_loss"] = fm_loss

    return total_loss, loss_dict


def compute_adaptive_weight(
    nll_loss: torch.Tensor,
    g_loss: torch.Tensor,
    last_layer: nn.Parameter,
    discriminator_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute adaptive discriminator weight (VQGAN-style).

    This balances the GAN loss contribution with the reconstruction loss by
    computing the ratio of their gradient norms with respect to the last
    decoder layer. This prevents the discriminator from dominating training.

    Reference: Esser et al., "Taming Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2012.09841

    Args:
        nll_loss: Reconstruction loss (MSE, L1, perceptual, etc.) - must have grad enabled
        g_loss: Generator's GAN loss - must have grad enabled
        last_layer: The last layer's weight parameter (e.g., decoder.final_conv.weight)
        discriminator_weight: Base discriminator weight to scale by

    Returns:
        Adaptive weight to multiply with the GAN loss
    """
    # Compute gradients of reconstruction loss w.r.t. last decoder layer
    nll_grads = torch.autograd.grad(
        nll_loss, last_layer, retain_graph=True, allow_unused=True
    )[0]

    # Compute gradients of GAN loss w.r.t. last decoder layer
    g_grads = torch.autograd.grad(
        g_loss, last_layer, retain_graph=True, allow_unused=True
    )[0]

    # Handle case where gradients are None (shouldn't happen in normal training)
    if nll_grads is None or g_grads is None:
        return torch.tensor(discriminator_weight, device=g_loss.device)

    # Compute adaptive weight as ratio of gradient norms
    # This ensures GAN gradients are scaled to match reconstruction gradient magnitude
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)

    # Clamp to prevent extreme values
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

    return discriminator_weight * d_weight
