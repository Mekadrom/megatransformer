from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.audio.shared_window_buffer import SharedWindowBuffer


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
