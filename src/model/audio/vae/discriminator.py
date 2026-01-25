import torch
import torch.nn as nn
import torch.nn.functional as F

from config.audio.vae.discriminator import MEL_COMBINED_DISCRIMINATOR_CONFIGS, MelDomainCombinedDiscriminatorConfig, MelDomainMultiPeriodDiscriminatorConfig, MelDomainMultiScaleDiscriminatorConfig


class MelDomainPeriodSubDiscriminator(nn.Module):
    """Sub-discriminator for a specific period in MelMultiPeriodDiscriminator."""
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        period: int = 2,
        n_layers: int = 4,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        self.period = period
        norm_f = nn.utils.spectral_norm if use_spectral_norm else lambda x: x

        self.layers = nn.ModuleList()
        # After reshape in forward, channels become in_channels * period
        channels = in_channels * period

        for i in range(n_layers):
            out_channels = min(base_channels * (2 ** i), 512)
            # Use asymmetric kernel: small in freq, larger in time
            kernel_size = (3, 5)
            stride = (2, 3) if i < n_layers - 1 else (1, 1)
            padding = (1, 2)

            layer = nn.Sequential(
                norm_f(nn.Conv2d(channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.layers.append(layer)
            channels = out_channels

        self.final = norm_f(nn.Conv2d(channels, 1, kernel_size=(3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: [B, 1, n_mels, T] mel spectrogram

        Returns:
            output: discriminator output
            features: intermediate features for feature matching
        """
        B, C, H, T = x.shape

        # Reshape to capture periodicity in time dimension
        # Pad time dimension to be divisible by period
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            # For 4D tensor, reflect mode needs (left, right, top, bottom) format
            x = F.pad(x, (0, pad_len, 0, 0), mode='reflect')
            T = T + pad_len

        # Reshape: [B, C, H, T] -> [B, C, H, T//period, period] -> [B, C*period, H, T//period]
        x = x.view(B, C, H, T // self.period, self.period)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, period, H, T//period]
        x = x.view(B, C * self.period, H, T // self.period)

        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        output = self.final(x)
        return output, features


class MelDomainMultiPeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator for mel spectrograms.

    Reshapes the mel spectrogram to analyze different periodicities in time,
    capturing harmonic structure at various scales.
    """
    def __init__(self, config: MelDomainMultiPeriodDiscriminatorConfig):
        super().__init__()

        self.config = config

        self.discriminators = nn.ModuleList([
            MelDomainPeriodSubDiscriminator(
                in_channels=config.in_channels,
                base_channels=config.base_channels,
                period=p,
                use_spectral_norm=config.use_spectral_norm,
            )
            for p in config.periods
        ])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs = []
        all_features = []

        for disc in self.discriminators:
            out, feats = disc(x)
            outputs.append(out)
            all_features.append(feats)

        return outputs, all_features


class MelDomainPatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for mel spectrograms.

    Uses asymmetric kernels and strides to handle non-square mel inputs.
    Input shape: [B, 1, n_mels, T] e.g. [B, 1, 80, 1875]
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        n_layers: int = 3,
        kernel_sizes: list = None,
        strides: list = None,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        # Default asymmetric kernels/strides for mel spectrograms
        if kernel_sizes is None:
            kernel_sizes = [(3, 5)] * n_layers + [(3, 5)]
        if strides is None:
            # Downsample more aggressively in time dimension
            strides = [(2, 3)] + [(2, 5)] * (n_layers - 1) + [(1, 1)]

        norm_f = nn.utils.spectral_norm if use_spectral_norm else lambda x: x

        # Build layers
        self.layers = nn.ModuleList()
        channels = in_channels

        for i in range(n_layers):
            out_channels = min(base_channels * (2 ** i), 512)
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)

            layer = nn.Sequential(
                norm_f(nn.Conv2d(channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.layers.append(layer)
            channels = out_channels

        # Final layer: single channel output
        final_kernel = kernel_sizes[-1]
        final_padding = (final_kernel[0] // 2, final_kernel[1] // 2)
        self.final_layer = norm_f(nn.Conv2d(channels, 1, kernel_size=final_kernel, stride=strides[-1], padding=final_padding))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: [B, 1, n_mels, T] mel spectrogram tensor

        Returns:
            output: [B, 1, H', W'] patch-wise predictions
            features: list of intermediate feature maps for feature matching
        """
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        output = self.final_layer(x)
        return output, features


class MelDomainMultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for mel spectrograms.

    Operates on different temporal resolutions to capture both
    fine phonetic details and longer-term structure.
    """
    def __init__(self, config: MelDomainMultiScaleDiscriminatorConfig):
        super().__init__()

        self.discriminators = nn.ModuleList([
            MelDomainPatchDiscriminator(
                in_channels=config.in_channels,
                base_channels=config.base_channels,
                n_layers=config.n_layers,
                use_spectral_norm=config.use_spectral_norm,
            )
            for _ in range(config.n_scales)
        ])

        # Downsample more in time than frequency
        self.downsample = nn.AvgPool2d(kernel_size=(2, 3), stride=(2, 3), padding=(0, 1))

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            x: [B, 1, n_mels, T] mel spectrogram tensor

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


class MelDomainCombinedDiscriminator(nn.Module):
    """
    Combined discriminator that uses multiple discriminator types together.

    Combines multi-scale (captures different temporal resolutions) and
    multi-period (captures harmonic periodicities) discriminators for
    comprehensive mel spectrogram discrimination.
    """
    def __init__(self, config: MelDomainCombinedDiscriminatorConfig):
        super().__init__()

        self.config = config

        self.discriminators = nn.ModuleList()

        if config.multi_scale_config is not None:
            self.discriminators.append(MelDomainMultiScaleDiscriminator(config=config.multi_scale_config))

        if config.multi_period_config is not None:
            self.discriminators.append(MelDomainMultiPeriodDiscriminator(config=config.multi_period_config))

        if len(self.discriminators) == 0:
            raise ValueError("At least one discriminator type must be enabled")

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "MelDomainCombinedDiscriminator":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = MelCombinedDiscriminator.from_config("small", multi_scale_config=MelMultiScaleDiscriminatorConfig(base_channels=64))
        """
        if config_name not in MEL_COMBINED_DISCRIMINATOR_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(MEL_COMBINED_DISCRIMINATOR_CONFIGS.keys())}")

        config = MEL_COMBINED_DISCRIMINATOR_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = MelDomainCombinedDiscriminatorConfig(**config_dict)

        return cls(config)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            x: [B, 1, n_mels, T] mel spectrogram tensor

        Returns:
            outputs: list of all discriminator outputs (flattened across discriminators)
            all_features: list of feature lists for feature matching
        """
        all_outputs = []
        all_features = []

        for disc in self.discriminators:
            outputs, features = disc(x)
            all_outputs.extend(outputs)
            all_features.extend(features)

        return all_outputs, all_features


class MelInstanceNoiseScheduler:
    """
    Scheduler for decaying instance noise over training.

    Starts with high noise to prevent shortcut learning, then decays
    to let the discriminator learn finer details.

    Usage:
        noise_scheduler = MelInstanceNoiseScheduler(initial_std=0.2, decay_steps=50000)
        for step in range(num_steps):
            noise_std = noise_scheduler.get_std(step)
            real_noisy = add_mel_instance_noise(real, noise_std)
            fake_noisy = add_mel_instance_noise(fake, noise_std)
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


def add_mel_instance_noise(
    mels: torch.Tensor,
    std: float = 0.1,
) -> torch.Tensor:
    """
    Add instance noise to mel spectrograms for discriminator regularization.

    This prevents the discriminator from finding shortcuts based on artifacts
    (blur, artifacts) and forces it to learn actual spectrogram structure.

    Args:
        mels: [B, 1, n_mels, T] mel spectrogram tensor
        std: Standard deviation of Gaussian noise (decay over training)

    Returns:
        Noisy mel spectrograms
    """
    if std <= 0:
        return mels
    return mels + std * torch.randn_like(mels)


def mel_discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """
    Discriminator hinge loss for mel spectrogram discriminators.
    Real samples should produce positive values, fake should produce negative.
    """
    loss = 0.0
    for real, fake in zip(disc_real_outputs, disc_fake_outputs):
        loss += torch.mean(F.relu(1 - real))
        loss += torch.mean(F.relu(1 + fake))
    return loss


def compute_mel_discriminator_loss(
    discriminator: nn.Module,
    real_mels: torch.Tensor,
    fake_mels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute discriminator loss for mel spectrogram discriminators.

    Args:
        discriminator: The discriminator module
        real_mels: Real mel spectrograms from the dataset [B, 1, n_mels, T]
        fake_mels: Fake mel spectrograms from the generator (detached)

    Returns:
        total_loss: Combined discriminator loss
        loss_dict: Dictionary with individual loss components
    """
    # Get discriminator outputs
    real_outputs, real_features = discriminator(real_mels)
    fake_outputs, fake_features = discriminator(fake_mels.detach())

    # Handle both single and multi-scale discriminators
    if not isinstance(real_outputs, list):
        real_outputs = [real_outputs]
        fake_outputs = [fake_outputs]

    d_loss = mel_discriminator_loss(real_outputs, fake_outputs)

    loss_dict = {
        "d_loss": d_loss,
        "d_real_mean": sum(r.mean() for r in real_outputs) / len(real_outputs),
        "d_fake_mean": sum(f.mean() for f in fake_outputs) / len(fake_outputs),
    }

    return d_loss, loss_dict


def r1_mel_gradient_penalty(
    real_mels: torch.Tensor,
    discriminator: nn.Module,
) -> torch.Tensor:
    """
    R1 gradient penalty for mel spectrogram discriminators (Mescheder et al., 2018).

    Penalizes the gradient norm on real mel spectrograms, encouraging the discriminator
    to have flat responses around real data. This stabilizes training and
    prevents the discriminator from focusing only on detecting fake artifacts.

    Args:
        real_mels: [B, 1, n_mels, T] real mel spectrogram tensor (will enable gradients)
        discriminator: The mel discriminator module

    Returns:
        R1 penalty (scalar tensor)
    """
    real_mels = real_mels.detach().requires_grad_(True)

    real_outputs, _ = discriminator(real_mels)

    # Handle multi-scale discriminators
    if isinstance(real_outputs, list):
        real_outputs = sum(r.sum() for r in real_outputs)
    else:
        real_outputs = real_outputs.sum()

    # Compute gradients
    grads = torch.autograd.grad(
        outputs=real_outputs,
        inputs=real_mels,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # R1 = E[||∇D(x)||²]
    penalty = grads.pow(2).reshape(grads.size(0), -1).sum(1).mean()

    return penalty


def mel_generator_loss(disc_fake_outputs: list[torch.Tensor]) -> torch.Tensor:
    """
    Generator hinge loss for mel spectrogram discriminators.
    Generator wants discriminator to output positive values for fake samples.
    """
    loss = 0.0
    for fake in disc_fake_outputs:
        loss += -torch.mean(fake)
    return loss


def mel_feature_matching_loss(
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


def compute_mel_generator_gan_loss(
    discriminator: nn.Module,
    real_mels: torch.Tensor,
    fake_mels: torch.Tensor,
    feature_matching_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute generator adversarial and feature matching losses for mel spectrograms.

    Args:
        discriminator: The discriminator module
        real_mels: Real mel spectrograms (for feature matching)
        fake_mels: Fake mel spectrograms from the generator (not detached)
        feature_matching_weight: Weight for feature matching loss

    Returns:
        total_loss: Combined generator GAN loss
        loss_dict: Dictionary with individual loss components
    """
    # Get discriminator outputs for fake mels
    fake_outputs, fake_features = discriminator(fake_mels)

    # Handle both single and multi-scale discriminators
    if not isinstance(fake_outputs, list):
        fake_outputs = [fake_outputs]
    if not isinstance(fake_features[0], list):
        fake_features = [fake_features]

    # Adversarial loss
    g_adv_loss = mel_generator_loss(fake_outputs)

    loss_dict = {
        "g_adv_loss": g_adv_loss,
    }

    total_loss = g_adv_loss

    # Feature matching loss (optional)
    if feature_matching_weight > 0:
        with torch.no_grad():
            real_outputs, real_features = discriminator(real_mels)
            if not isinstance(real_features[0], list):
                real_features = [real_features]

        fm_loss = mel_feature_matching_loss(real_features, fake_features)
        total_loss = total_loss + feature_matching_weight * fm_loss
        loss_dict["g_fm_loss"] = fm_loss

    return total_loss, loss_dict
