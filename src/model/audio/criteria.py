from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


from utils.audio_utils import SharedWindowBuffer, configurable_mel_spectrogram


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for better audio quality.
    
    This loss computes the L1 loss between the STFT of the predicted and
    ground truth waveforms at multiple resolutions, which helps capture
    both fine and coarse time-frequency structures.
    """
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        fft_sizes: list[int] = [256, 512, 1024, 2048],
        hop_sizes: list[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes)
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        
        self.buffer_lookup = {}

        # Create window buffers
        for s in set(fft_sizes):
            self.register_buffer(f"window_{s}", shared_window_buffer.get_window(s, torch.device('cpu')))
            self.buffer_lookup[str(s)] = getattr(self, f"window_{s}")
    
    def stft_magnitude(
        self,
        x: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_length: int,
    ) -> torch.Tensor:
        """Calculate STFT magnitude."""

        # stft requires float32 input
        x_float32 = x.float()
        x_stft = torch.stft(
            x_float32.squeeze(1),
            fft_size,
            hop_size,
            win_length,
            self.buffer_lookup[str(win_length)].to(x_float32.dtype).to(x_float32.device),
            return_complex=True
        )
        return torch.abs(x_stft).to(x.dtype)
    
    def complex_stft_loss(self, pred, target_complex_stft, fft_size, hop_size, win_length) -> torch.Tensor:
        pred_stft = torch.stft(
            pred.float(), fft_size, hop_size, win_length,
            self.buffer_lookup[str(win_length)].to(pred.dtype).to(pred.device),
            return_complex=True
            )
        return F.l1_loss(pred_stft.real, target_complex_stft.real) + F.l1_loss(pred_stft.imag, target_complex_stft.imag)

    def forward(
        self, 
        pred_waveform: torch.Tensor, 
        target_waveform: torch.Tensor,
        pred_stft: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate multi-resolution STFT loss.
        
        Args:
            pred_waveform: [B, 1, T] Predicted waveform
            target_waveform: [B, 1, T] Target waveform
            pred_stft: Optional[torch.Tensor] = None Precomputed STFT of predicted waveform (for models that directly predict STFT for iSTFT conversion to waveforms)
            
        Returns:
            Tuple of (sc_loss, mag_loss) - spectral convergence and magnitude losses
        """
        sc_loss = 0.0
        mag_loss = 0.0
        complex_stft_loss = 0.0
        
        for fft_size, hop_size in zip(
            self.fft_sizes, self.hop_sizes
        ):
            pred_mag = self.stft_magnitude(
                pred_waveform, fft_size, hop_size, fft_size
            )
            target_mag = self.stft_magnitude(
                target_waveform, fft_size, hop_size, fft_size
            )
            
            # Spectral convergence loss
            target_norm = torch.norm(target_mag, p="fro").clamp(min=0.1)
            sc_loss += torch.norm(target_mag - pred_mag, p="fro") / target_norm
            
            # Log magnitude loss
            log_pred_mag = torch.log(pred_mag.clamp(min=1e-5))
            log_target_mag = torch.log(target_mag.clamp(min=1e-5))
            mag_loss += F.l1_loss(log_pred_mag, log_target_mag)

            # Complex STFT loss
            complex_stft_loss += self.complex_stft_loss(
                pred_waveform.squeeze(1),
                torch.stft(
                    target_waveform.squeeze(1).to(torch.float32), fft_size, hop_size,
                    window=self.buffer_lookup[str(fft_size)].to(torch.float32).to(target_waveform.device), return_complex=True
                ),
                fft_size,
                hop_size,
                fft_size
            ).to(pred_waveform.dtype)
        
        # Normalize by number of STFT resolutions
        sc_loss = sc_loss / len(self.fft_sizes)
        mag_loss = mag_loss / len(self.fft_sizes)
        complex_stft_loss = complex_stft_loss / len(self.fft_sizes)
        return sc_loss, mag_loss, complex_stft_loss


class MultiScaleMelLoss(nn.Module):
    """
    Multi-scale mel spectrogram loss for better frequency coverage.

    Computes mel spectrograms at multiple hop lengths and n_fft sizes,
    providing different time-frequency tradeoffs:
    - Small hop_length: Better time resolution (transients, attacks)
    - Large hop_length: Better frequency resolution (harmonics, tones)

    This helps the vocoder learn to reproduce both transients and
    steady-state harmonics accurately.
    """
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        sample_rate: int = 16000,
        n_mels: int = 80,
        scales: list[tuple[int, int]] = None,  # List of (n_fft, hop_length) pairs
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        # Default scales: different time-frequency tradeoffs
        if scales is None:
            scales = [
                (512, 64),    # Fine time resolution
                (1024, 128),  # Balanced
                (1024, 256),  # Default
                (2048, 512),  # Fine frequency resolution
            ]

        self.scales = scales
        self.shared_window_buffer = shared_window_buffer

        # Pre-compute mel filterbanks for each scale
        for i, (n_fft, hop_length) in enumerate(scales):
            mel_fb = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
                sample_rate=sample_rate,
            )
            self.register_buffer(f'mel_fb_{i}', mel_fb)
            self.register_buffer(f'window_{i}', shared_window_buffer.get_window(n_fft, torch.device('cpu')))

    def forward(self, pred_waveform: torch.Tensor, target_waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale mel loss between predicted and target waveforms.

        Args:
            pred_waveform: [B, T] predicted waveform
            target_waveform: [B, T] target waveform

        Returns:
            Scalar loss value (mean across all scales)
        """
        orig_dtype = pred_waveform.dtype
        total_loss = 0.0

        for i, (n_fft, hop_length) in enumerate(self.scales):
            window = getattr(self, f'window_{i}').to(pred_waveform.device)

            # Compute mel for prediction
            pred_mel, _ = configurable_mel_spectrogram(
                audio=pred_waveform,
                sample_rate=self.sample_rate,
                hop_length=hop_length,
                win_length=n_fft,
                n_mels=self.n_mels,
                n_fft=n_fft,
                f_min=self.f_min,
                f_max=self.f_max,
                power=1,
                normalized=False,
                min_max_energy_norm=False,
                norm="slaney",
                mel_scale="slaney",
                compression=False,
                window_provider=lambda win_size, w=window: w,
            )

            # Compute mel for target
            target_mel, _ = configurable_mel_spectrogram(
                audio=target_waveform,
                sample_rate=self.sample_rate,
                hop_length=hop_length,
                win_length=n_fft,
                n_mels=self.n_mels,
                n_fft=n_fft,
                f_min=self.f_min,
                f_max=self.f_max,
                power=1,
                normalized=False,
                min_max_energy_norm=False,
                norm="slaney",
                mel_scale="slaney",
                compression=False,
                window_provider=lambda win_size, w=window: w,
            )

            pred_mel = pred_mel.to(orig_dtype)
            target_mel = target_mel.to(orig_dtype)

            # Log mel with clamping for numerical stability
            pred_log_mel = torch.log(pred_mel.clamp(min=1e-5))
            target_log_mel = torch.log(target_mel.clamp(min=1e-5))

            # Match lengths (different hop lengths = different time dimensions)
            min_len = min(pred_log_mel.shape[-1], target_log_mel.shape[-1])
            total_loss = total_loss + F.l1_loss(
                pred_log_mel[..., :min_len],
                target_log_mel[..., :min_len]
            )

        return total_loss / len(self.scales)
