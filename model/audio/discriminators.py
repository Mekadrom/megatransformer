import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from utils.audio_utils import SharedWindowBuffer


class PeriodDiscriminator(nn.Module):
    """Single period discriminator that reshapes audio into 2D based on period."""
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            nn.utils.spectral_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = nn.utils.spectral_norm(nn.Conv2d(256, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []

        # Reshape: [B, T] -> [B, 1, T//period, period]
        b, t = x.shape
        pad = (self.period - (t % self.period)) % self.period
        if pad > 0:
            x = F.pad(x, (0, pad), mode='reflect')
        x = x.view(b, 1, -1, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)

        return x.flatten(1, -1), features

class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator using prime periods to capture different periodic structures."""
    def __init__(self, periods: list[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs = []
        all_features = []
        for d in self.discriminators:
            out, feats = d(x)
            outputs.append(out)
            all_features.append(feats)
        return outputs, all_features


class ScaleDiscriminator(nn.Module):
    """Single scale discriminator operating on raw or downsampled audio."""
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.parametrizations.weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 64, 15, 1, padding=7)),
            norm_f(nn.Conv1d(64, 64, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 128, 41, 2, groups=8, padding=20)),
            norm_f(nn.Conv1d(128, 128, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(128, 128, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(128, 256, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(256, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []

        # Add channel dimension: [B, T] -> [B, 1, T]
        x = x.unsqueeze(1)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)

        return x.flatten(1, -1), features


class PhaseAwareSpectrogramDiscriminator(nn.Module):
    """
    Single spectrogram discriminator operating on STFT magnitudes.
    Uses 2D convolutions on the [frequency, time] spectrogram.
    """
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        channels: list[int] = [16, 32, 64, 128, 256],
        kernel_size: tuple[int, int] = (3, 9),  # (freq, time)
        stride: tuple[int, int] = (1, 2),
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Register window buffer
        self.register_buffer('window', shared_window_buffer.get_window(win_length, torch.device('cpu')))
        
        # Input channels = 2 (real and imaginary parts)
        in_ch = 2
        self.convs = nn.ModuleList()
        
        for out_ch in channels:
            self.convs.append(
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv2d(
                            in_ch, out_ch,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
                        )
                    ),
                    nn.LeakyReLU(0.1, inplace=True)
                )
            )
            in_ch = out_ch
        
        # Final conv to single channel
        self.conv_post = nn.utils.spectral_norm(
            nn.Conv2d(channels[-1], 1, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: [B, T] waveform
        Returns:
            output: [B, N] flattened discriminator output
            features: list of intermediate feature maps for feature matching
        """
        # Compute STFT magnitude
        # x: [B, T] -> spec: [B, F, T']
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        x = torch.stack([spec.real, spec.imag], dim=1)
        
        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        
        out = self.conv_post(x)
        features.append(out)
        
        return out.flatten(1, -1), features


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator operating at different audio resolutions."""
    def __init__(self, n_scales: int = 3):
        super().__init__()

        self.discriminators = nn.ModuleList()
        self.pooling = nn.ModuleList()
        for i in range(n_scales):
            use_spectral_norm = (i == 0)  # Only use spectral norm for the first scale
            self.discriminators.append(ScaleDiscriminator(use_spectral_norm=use_spectral_norm))
            if i < n_scales - 1:
                self.pooling.append(nn.AvgPool1d(4, 2, padding=2))

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs = []
        all_features = []

        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling[i - 1](x)
            out, feats = d(x)
            outputs.append(out)
            all_features.append(feats)

        return outputs, all_features


class MultiResolutionSpectrogramDiscriminator(nn.Module):
    """
    Multi-resolution spectrogram discriminator (MRSD).
    Analyzes audio at multiple STFT resolutions to capture both
    fine temporal detail and broad frequency patterns.
    """
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        resolutions: list[tuple[int, int, int]] = [
            (1024, 256, 1024),   # (n_fft, hop_length, win_length) - balanced
            (2048, 512, 2048),   # Low frequency detail, coarse time
            (512, 128, 512),     # High frequency detail, fine time
        ],
        channels: list[int] = [16, 32, 64, 128, 256],
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PhaseAwareSpectrogramDiscriminator(
                shared_window_buffer=shared_window_buffer,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                channels=channels,
            )
            for n_fft, hop, win in resolutions
        ])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            x: [B, T] waveform
        Returns:
            outputs: list of discriminator outputs
            all_features: list of feature lists for feature matching
        """
        outputs = []
        all_features = []
        
        for disc in self.discriminators:
            out, feats = disc(x)
            outputs.append(out)
            all_features.append(feats)
        
        return outputs, all_features


class CombinedDiscriminator(nn.Module):
    """Combined MPD + MSD + MRSD discriminator."""
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        mpd_periods: list[int] = [2, 3, 5, 7, 11, 13, 17],
        n_msd_scales: int = 4,
        mrsd_resolutions: list[tuple[int, int, int]] = [
            (1024, 256, 1024),
            (2048, 512, 2048),
            (512, 128, 512),
        ],
        mrsd_channels: list[int] = [16, 32, 64, 128, 256],
    ):
        super().__init__()

        self.mpd = MultiPeriodDiscriminator(periods=mpd_periods)
        self.msd = MultiScaleDiscriminator(n_msd_scales)
        self.mrsd = MultiResolutionSpectrogramDiscriminator(shared_window_buffer=shared_window_buffer, resolutions=mrsd_resolutions, channels=mrsd_channels)
    def forward(self, x: torch.Tensor) -> dict[str, tuple[list[torch.Tensor], list[list[torch.Tensor]]]]:
        """
        Args:
            x: [B, T] waveform
        Returns:
            Dictionary with outputs and features from each discriminator type
        """
        results = {
            "mpd": self.mpd(x),
            "msd": self.msd(x),
            "mrsd": self.mrsd(x),
        }
        return results


def small_combined_disc(shared_window_buffer: SharedWindowBuffer) -> CombinedDiscriminator:
    """Creates a CombinedDiscriminator with updated configuration."""
    return CombinedDiscriminator(
        shared_window_buffer=shared_window_buffer,
        mpd_periods=[2, 3, 5, 7, 11, 13, 17],
        n_msd_scales=4,
        mrsd_resolutions=[
            (1024, 256, 1024),
            (2048, 512, 2048),
            (512, 128, 512),
            (256, 64, 256),
        ],
        mrsd_channels=[8, 16, 32, 48, 64, 96],
    )

model_config_lookup = {
    "small_combined_disc": small_combined_disc,
}


# =============================================================================
# Mel Spectrogram Discriminators (for VAE-GAN training on mel spectrograms)
# =============================================================================

class MelPatchDiscriminator(nn.Module):
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


class MelMultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for mel spectrograms.

    Operates on different temporal resolutions to capture both
    fine phonetic details and longer-term structure.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        n_layers: int = 3,
        n_scales: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            MelPatchDiscriminator(
                in_channels=in_channels,
                base_channels=base_channels,
                n_layers=n_layers,
                use_spectral_norm=use_spectral_norm,
            )
            for _ in range(n_scales)
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


class MelMultiPeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator for mel spectrograms.

    Reshapes the mel spectrogram to analyze different periodicities in time,
    capturing harmonic structure at various scales.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        periods: list = None,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        if periods is None:
            periods = [2, 3, 5, 7, 11]

        self.periods = periods
        self.discriminators = nn.ModuleList([
            MelPeriodSubDiscriminator(
                in_channels=in_channels,
                base_channels=base_channels,
                period=p,
                use_spectral_norm=use_spectral_norm,
            )
            for p in periods
        ])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs = []
        all_features = []

        for disc in self.discriminators:
            out, feats = disc(x)
            outputs.append(out)
            all_features.append(feats)

        return outputs, all_features


class MelPeriodSubDiscriminator(nn.Module):
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
        channels = in_channels

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
            x = F.pad(x, (0, pad_len), mode='reflect')
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


class MelCombinedDiscriminator(nn.Module):
    """
    Combined discriminator that uses multiple discriminator types together.

    Combines multi-scale (captures different temporal resolutions) and
    multi-period (captures harmonic periodicities) discriminators for
    comprehensive mel spectrogram discrimination.
    """
    def __init__(
        self,
        # Multi-scale discriminator settings
        use_multi_scale: bool = True,
        multi_scale_base_channels: int = 32,
        multi_scale_n_layers: int = 3,
        multi_scale_n_scales: int = 2,
        # Multi-period discriminator settings
        use_multi_period: bool = True,
        multi_period_base_channels: int = 32,
        multi_period_periods: list = None,
        # Shared settings
        in_channels: int = 1,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList()

        if use_multi_scale:
            self.discriminators.append(
                MelMultiScaleDiscriminator(
                    in_channels=in_channels,
                    base_channels=multi_scale_base_channels,
                    n_layers=multi_scale_n_layers,
                    n_scales=multi_scale_n_scales,
                    use_spectral_norm=use_spectral_norm,
                )
            )

        if use_multi_period:
            self.discriminators.append(
                MelMultiPeriodDiscriminator(
                    in_channels=in_channels,
                    base_channels=multi_period_base_channels,
                    periods=multi_period_periods,
                    use_spectral_norm=use_spectral_norm,
                )
            )

        if len(self.discriminators) == 0:
            raise ValueError("At least one discriminator type must be enabled")

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


# Mel spectrogram discriminator loss functions

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


# Mel spectrogram discriminator model configs

def tiny_mel_patch_discriminator() -> MelPatchDiscriminator:
    """Tiny patch discriminator for mel spectrograms (~50K params)."""
    return MelPatchDiscriminator(
        in_channels=1,
        base_channels=16,
        n_layers=2,
        use_spectral_norm=True,
    )


def mini_mel_patch_discriminator() -> MelPatchDiscriminator:
    """Mini patch discriminator for mel spectrograms (~200K params)."""
    return MelPatchDiscriminator(
        in_channels=1,
        base_channels=32,
        n_layers=3,
        use_spectral_norm=True,
    )


def small_mel_patch_discriminator() -> MelPatchDiscriminator:
    """Small patch discriminator for mel spectrograms (~800K params)."""
    return MelPatchDiscriminator(
        in_channels=1,
        base_channels=64,
        n_layers=3,
        use_spectral_norm=True,
    )


def tiny_mel_multi_scale_discriminator() -> MelMultiScaleDiscriminator:
    """Tiny multi-scale discriminator for mel spectrograms (~100K params)."""
    return MelMultiScaleDiscriminator(
        in_channels=1,
        base_channels=16,
        n_layers=2,
        n_scales=2,
        use_spectral_norm=True,
    )


def mini_mel_multi_scale_discriminator() -> MelMultiScaleDiscriminator:
    """Mini multi-scale discriminator for mel spectrograms (~400K params)."""
    return MelMultiScaleDiscriminator(
        in_channels=1,
        base_channels=32,
        n_layers=3,
        n_scales=2,
        use_spectral_norm=True,
    )


def mel_multi_scale_discriminator() -> MelMultiScaleDiscriminator:
    """Standard multi-scale discriminator for mel spectrograms (~2.5M params)."""
    return MelMultiScaleDiscriminator(
        in_channels=1,
        base_channels=64,
        n_layers=3,
        n_scales=3,
        use_spectral_norm=True,
    )


def mel_multi_period_discriminator() -> MelMultiPeriodDiscriminator:
    """Multi-period discriminator for mel spectrograms (~1.5M params)."""
    return MelMultiPeriodDiscriminator(
        in_channels=1,
        base_channels=32,
        periods=[2, 3, 5, 7, 11],
        use_spectral_norm=True,
    )


def small_mel_multi_scale_discriminator() -> MelMultiScaleDiscriminator:
    """Small multi-scale discriminator for mel spectrograms (~1.2M params)."""
    return MelMultiScaleDiscriminator(
        in_channels=1,
        base_channels=48,
        n_layers=4,
        n_scales=3,
        use_spectral_norm=True,
    )


def mini_mel_combined_discriminator() -> MelCombinedDiscriminator:
    """
    Mini combined discriminator (~700K params).
    Combines multi-scale and multi-period for comprehensive discrimination.
    """
    return MelCombinedDiscriminator(
        use_multi_scale=True,
        multi_scale_base_channels=24,
        multi_scale_n_layers=3,
        multi_scale_n_scales=2,
        use_multi_period=True,
        multi_period_base_channels=24,
        multi_period_periods=[2, 3, 5, 7],
        use_spectral_norm=True,
    )


def small_mel_combined_discriminator() -> MelCombinedDiscriminator:
    """
    Small combined discriminator (~1.2M params).
    Combines multi-scale and multi-period for comprehensive discrimination.
    """
    return MelCombinedDiscriminator(
        use_multi_scale=True,
        multi_scale_base_channels=32,
        multi_scale_n_layers=3,
        multi_scale_n_scales=2,
        use_multi_period=True,
        multi_period_base_channels=32,
        multi_period_periods=[2, 3, 5, 7, 11],
        use_spectral_norm=True,
    )


def mel_combined_discriminator() -> MelCombinedDiscriminator:
    """
    Standard combined discriminator (~2.5M params).
    Combines multi-scale and multi-period for comprehensive discrimination.
    """
    return MelCombinedDiscriminator(
        use_multi_scale=True,
        multi_scale_base_channels=48,
        multi_scale_n_layers=4,
        multi_scale_n_scales=3,
        use_multi_period=True,
        multi_period_base_channels=48,
        multi_period_periods=[2, 3, 5, 7, 11],
        use_spectral_norm=True,
    )


mel_discriminator_config_lookup = {
    "tiny_patch": tiny_mel_patch_discriminator,
    "mini_patch": mini_mel_patch_discriminator,
    "small_patch": small_mel_patch_discriminator,
    "tiny_multi_scale": tiny_mel_multi_scale_discriminator,
    "mini_multi_scale": mini_mel_multi_scale_discriminator,
    "small_multi_scale": small_mel_multi_scale_discriminator,
    "multi_scale": mel_multi_scale_discriminator,
    "multi_period": mel_multi_period_discriminator,
    "mini_combined": mini_mel_combined_discriminator,
    "small_combined": small_mel_combined_discriminator,
    "combined": mel_combined_discriminator,
}
