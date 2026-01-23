import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional

from config.audio.vocoder.discriminator import DISCRIMINATOR_CONFIGS, WaveformDomainDiscriminatorConfig, WaveformDomainMultiPeriodDiscriminatorConfig, WaveformDomainMultiResolutionDiscriminatorConfig, WaveformDomainMultiResolutionDiscriminatorConfig, WaveformDomainMultiScaleDiscriminatorConfig
from utils.audio_utils import SharedWindowBuffer


class PeriodDiscriminator(nn.Module):
    """Single period discriminator that reshapes audio into 2D based on period."""
    def __init__(self, period: int, base_channels: int = 32, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            nn.utils.spectral_norm(nn.Conv2d(1, base_channels, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = nn.utils.spectral_norm(nn.Conv2d(base_channels * 8, 1, (3, 1), 1, padding=(1, 0)))
    
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
    def __init__(self, config: WaveformDomainMultiPeriodDiscriminatorConfig):
        super().__init__()

        self.config = config

        if config.base_channels is None:
            config.base_channels = [32] * len(config.periods)

        assert len(config.base_channels) == len(config.periods), "base_channels length must match periods length"

        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p, base_channels=bc) for p, bc in zip(config.periods, config.base_channels)
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
    def __init__(self, base_channels: int = 64, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.parametrizations.weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, base_channels, 15, 1, padding=7)),
            norm_f(nn.Conv1d(base_channels, base_channels, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(base_channels, base_channels * 2, 41, 2, groups=8, padding=20)),
            norm_f(nn.Conv1d(base_channels * 2, base_channels * 2, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(base_channels * 2, base_channels * 2, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(base_channels * 2, base_channels * 4, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(base_channels * 4, 1, 3, 1, padding=1))

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
        channels: list[int] = [16, 32, 64, 128, 256],
        kernel_size: tuple[int, int] = (3, 9),  # (freq, time)
        stride: tuple[int, int] = (1, 2),
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Register window buffer
        self.register_buffer('window', shared_window_buffer.get_window(n_fft, torch.device('cpu')))
        
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
            win_length=self.n_fft,
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
    def __init__(self, config: WaveformDomainMultiScaleDiscriminatorConfig):
        super().__init__()

        self.config = config

        self.discriminators = nn.ModuleList()
        self.pooling = nn.ModuleList()
        for i, base_ch in enumerate(config.base_channels):
            use_spectral_norm = (i == 0)  # Only use spectral norm for the first scale
            self.discriminators.append(ScaleDiscriminator(base_channels=base_ch, use_spectral_norm=use_spectral_norm))
            if i < len(config.base_channels) - 1:
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
    def __init__(self, shared_window_buffer: SharedWindowBuffer, config: WaveformDomainMultiResolutionDiscriminatorConfig):
        super().__init__()

        self.config = config

        self.discriminators = nn.ModuleList([
            PhaseAwareSpectrogramDiscriminator(
                shared_window_buffer=shared_window_buffer,
                n_fft=n_fft,
                hop_length=hop,
                channels=config.base_channels,
            )
            for n_fft, hop in config.resolutions
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


class WaveformDomainDiscriminator(nn.Module):
    """Combined MPD + MSD + MRSD discriminator."""
    def __init__(self, shared_window_buffer: SharedWindowBuffer, config: WaveformDomainDiscriminatorConfig):
        super().__init__()

        self.config = config

        self.mpd = MultiPeriodDiscriminator(config.mpd_config)
        self.msd = MultiScaleDiscriminator(config.msd_config)
        self.mrsd = MultiResolutionSpectrogramDiscriminator(shared_window_buffer, config.mrsd_config)

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "WaveformDomainDiscriminator":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = WaveformDomainDiscriminator.from_config("small", mpd_config=custom_mpd_config)
        """
        if config_name not in DISCRIMINATOR_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(DISCRIMINATOR_CONFIGS.keys())}")

        config = DISCRIMINATOR_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = WaveformDomainDiscriminatorConfig(**config_dict)

        return cls(config)

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
