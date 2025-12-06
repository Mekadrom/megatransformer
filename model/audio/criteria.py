import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import megatransformer_utils

from model.audio import configurable_mel_spectrogram
from typing import Optional

from model.audio.shared_window_buffer import SharedWindowBuffer


# Multi-Resolution STFT Loss for better vocoder training
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
        win_lengths: list[int] = [256, 512, 1024, 2048],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
        self.buffer_lookup = {}

        # Create window buffers
        for s in set(win_lengths):
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
        
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            pred_mag = self.stft_magnitude(
                pred_waveform, fft_size, hop_size, win_length
            )
            target_mag = self.stft_magnitude(
                target_waveform, fft_size, hop_size, win_length
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
                    window=self.buffer_lookup[str(win_length)].to(torch.float32).to(target_waveform.device), return_complex=True
                ),
                fft_size,
                hop_size,
                win_length
            ).to(pred_waveform.dtype)
        
        # Normalize by number of STFT resolutions
        sc_loss = sc_loss / len(self.fft_sizes)
        mag_loss = mag_loss / len(self.fft_sizes)
        complex_stft_loss = complex_stft_loss / len(self.fft_sizes)
        return sc_loss, mag_loss, complex_stft_loss


class AudioGenerationLoss(nn.Module):
    """
    Combined loss function for training both diffusion model and vocoder.
    """
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        diffusion_loss_weight: float = 1.0,
        mel_loss_weight: float = 10.0,
        waveform_loss_weight: float = 1.0,
        stft_loss_weight: float = 2.0,
    ):
        super().__init__()
        self.diffusion_loss_weight = diffusion_loss_weight
        self.mel_loss_weight = mel_loss_weight
        self.waveform_loss_weight = waveform_loss_weight
        self.stft_loss_weight = stft_loss_weight
        
        self.stft_loss = MultiResolutionSTFTLoss(shared_window_buffer)
    
    def forward(
        self,
        pred_noise: torch.Tensor,
        noise: torch.Tensor,
        pred_mel: Optional[torch.Tensor] = None,
        target_mel: Optional[torch.Tensor] = None,
        pred_waveform: Optional[torch.Tensor] = None,
        target_waveform: Optional[torch.Tensor] = None,
        target_complex_stft: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculate the combined loss.
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        losses = {}
        
        # Diffusion loss (noise prediction)
        diffusion_loss = F.mse_loss(pred_noise, noise)
        losses["diffusion"] = diffusion_loss
        
        total_loss = self.diffusion_loss_weight * diffusion_loss
        
        # Optional mel loss
        if pred_mel is not None and target_mel is not None:
            mel_loss = F.l1_loss(pred_mel, target_mel)
            losses["mel"] = mel_loss
            total_loss = total_loss + self.mel_loss_weight * mel_loss
        
        # Optional waveform and STFT losses
        if pred_waveform is not None and target_waveform is not None:
            # Direct waveform loss
            waveform_loss = F.l1_loss(pred_waveform, target_waveform)
            losses["waveform"] = waveform_loss
            total_loss = total_loss + self.waveform_loss_weight * waveform_loss
            
            # Multi-resolution STFT loss
            sc_loss, mag_loss, complex_stft_loss = self.stft_loss(pred_waveform, target_waveform)
            losses["sc"] = sc_loss
            losses["mag"] = mag_loss
            losses["complex_stft"] = complex_stft_loss
            total_loss = total_loss + self.stft_loss_weight * (sc_loss + mag_loss + complex_stft_loss)
        
        return total_loss, losses

class StableMelSpectrogramLoss(nn.Module):
    def __init__(self, shared_window_buffer: SharedWindowBuffer, sample_rate, n_fft, hop_length, n_mels, mel_recon_loss_weight_linspace_max: float = 1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_recon_loss_weight_linspace_max = mel_recon_loss_weight_linspace_max
        
        # Pre-compute mel filterbank (no grad needed)
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.register_buffer('mel_fb', mel_fb)
        self.register_buffer('window', shared_window_buffer.get_window(1024, torch.device('cpu')))
    
    def weighted_mel_loss(self, pred_mel, target_mel, low_freq_weight=2.0, cutoff_bin=20):
        """Weight low frequency bins more heavily."""
        weights = torch.ones(pred_mel.shape[-2], device=pred_mel.device)
        weights[:cutoff_bin] = low_freq_weight
        weights = weights.view(1, -1, 1)  # (1, n_mels, 1)
        
        return (F.l1_loss(pred_mel, target_mel, reduction='none') * weights).mean()

    def forward(self, pred_waveform, target_log_mel):
        orig_dtype = pred_waveform.dtype
        # STFT
        mel_spec, _ = configurable_mel_spectrogram(
            audio=pred_waveform,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            win_length=1024,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            f_min=0.0,
            f_max=8000.0,
            power=1,
            normalized=False,
            min_max_energy_norm=False,
            norm="slaney",
            mel_scale="slaney",
            compression=False,
            window_provider=lambda win_size: self.window.to(pred_waveform.device)
        )
        mel_spec = mel_spec.to(orig_dtype)
        
        # Log with clamp
        log_mel = torch.log(mel_spec.clamp(min=1e-5))
        
        # Match lengths
        min_len = min(log_mel.shape[-1], target_log_mel.shape[-1])
        
        # return self.weighted_mel_loss(log_mel[..., :min_len], target_log_mel[..., :min_len])
        return F.l1_loss(log_mel[..., :min_len], target_log_mel[..., :min_len])


class PhaseLoss(nn.Module):
    def __init__(self, shared_window_buffer: SharedWindowBuffer, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.n_fft = config.audio_n_fft
        self.hop_length = config.audio_hop_length
        self.register_buffer('window', shared_window_buffer.get_window(self.n_fft, torch.device('cpu')))
        
    def anti_wrap(self, x):
        """Map phase difference to [-pi, pi]"""
        return torch.atan2(torch.sin(x), torch.cos(x))

    def forward(self, pred_wav, target_wav, target_complex_stfts, precomputed_stft: Optional[torch.Tensor] = None):
        orig_dtype = pred_wav.dtype
        pred_wav = pred_wav.to(torch.float32)

        if precomputed_stft is None:
            pred_stft = torch.stft(pred_wav, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        else:
            pred_stft = precomputed_stft
        
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_complex_stfts)
        
        # Energy mask - ignore silent regions
        target_mag = target_complex_stfts.abs()
        energy_threshold = target_mag.max() * 0.01  # Bottom 1% = silence
        mask = (target_mag > energy_threshold).float()
        
        # Match lengths
        min_t = min(pred_phase.shape[-1], target_phase.shape[-1])
        pred_phase = pred_phase[..., :min_t]
        target_phase = target_phase[..., :min_t]
        mask = mask[..., :min_t]
        
        # IP loss - masked
        phase_diff = self.anti_wrap(pred_phase - target_phase)
        ip_loss = (phase_diff.abs() * mask).sum() / (mask.sum() + 1e-8)
        
        # IAF loss - masked
        pred_iaf = torch.diff(pred_phase, dim=-1)
        target_iaf = torch.diff(target_phase, dim=-1)
        iaf_diff = self.anti_wrap(pred_iaf - target_iaf)
        mask_iaf = mask[..., 1:]  # Diff reduces length by 1
        iaf_loss = (iaf_diff.abs() * mask_iaf).sum() / (mask_iaf.sum() + 1e-8)
        
        # GD loss - masked  
        pred_gd = torch.diff(pred_phase, dim=-2)
        target_gd = torch.diff(target_phase, dim=-2)
        gd_diff = self.anti_wrap(pred_gd - target_gd)
        mask_gd = mask[..., 1:, :]  # Diff reduces freq dim by 1
        gd_loss = (gd_diff.abs() * mask_gd).sum() / (mask_gd.sum() + 1e-8)
        
        return ip_loss, iaf_loss, gd_loss


class HighFreqSTFTLoss(nn.Module):
    def __init__(self, shared_window_buffer: SharedWindowBuffer, n_fft, hop_length, cutoff_bin=256):
        super().__init__()
        self.n_fft = n_fft
        self.cutoff_bin = cutoff_bin
        self.hop_length = hop_length
        self.register_buffer('window', shared_window_buffer.get_window(n_fft, torch.device('cpu')))

    def forward(self, pred_wav, target_wav, target_complex_stfts, precomputed_stft: Optional[torch.Tensor] = None):
        pred_wav = pred_wav.to(torch.float32)
        target_wav = target_wav.to(torch.float32)

        if precomputed_stft is None:
            pred_stft = torch.stft(pred_wav, self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        else:
            pred_stft = precomputed_stft
        
        # make sure pred_stft and target_complex_stfts have the same shape
        min_frames = min(pred_stft.shape[-1], target_complex_stfts.shape[-1])
        pred_stft = pred_stft[..., :min_frames]
        target_complex_stfts = target_complex_stfts[..., :min_frames]

        pred_hf = pred_stft[..., self.cutoff_bin:, :].abs()
        target_hf = target_complex_stfts[..., self.cutoff_bin:, :].abs()

        weights = target_hf / (target_hf.sum(dim=-2, keepdim=True) + 1e-5)

        log_target_hf = torch.log(target_hf.clamp(min=1e-7))
        log_pred_hf = torch.log(pred_hf.clamp(min=1e-7))
        return F.l1_loss(log_pred_hf, log_target_hf)


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """
    Discriminator loss: real samples should be classified as 1, fake as 0.
    Uses least-squares GAN loss.
    """
    loss = 0.0
    for dr, df in zip(disc_real_outputs, disc_fake_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        f_loss = torch.mean(df ** 2)
        loss += r_loss + f_loss
    return loss


def generator_loss(disc_fake_outputs: list[torch.Tensor]) -> torch.Tensor:
    """
    Generator loss: fake samples should be classified as 1 (fool discriminator).
    Uses least-squares GAN loss.
    """
    loss = 0.0
    for df in disc_fake_outputs:
        loss += torch.mean((1 - df) ** 2)
    return loss


def feature_matching_loss(
    disc_real_features: list[list[torch.Tensor]],
    disc_fake_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    Feature matching loss with proper normalization.
    """
    loss = 0.0
    num_layers = 0
    
    for real_feats, fake_feats in zip(disc_real_features, disc_fake_features):
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            # Normalize by feature magnitude to make loss scale-invariant
            loss += F.l1_loss(fake_feat, real_feat.detach()) / (real_feat.detach().abs().mean() + 1e-5)
            num_layers += 1
    
    # Average over layers
    return loss / num_layers if num_layers > 0 else loss


def compute_discriminator_losses(
    disc_outputs_real: dict[str, tuple[list[torch.Tensor], list[list[torch.Tensor]]]],
    disc_outputs_fake: dict[str, tuple[list[torch.Tensor], list[list[torch.Tensor]]]],
) -> dict[str, torch.Tensor]:
    """Compute discriminator losses for all discriminator types."""
    losses = {}
    
    for key in disc_outputs_real:
        real_outs, _ = disc_outputs_real[key]
        fake_outs, _ = disc_outputs_fake[key]
        loss = discriminator_loss(real_outs, fake_outs)
        losses[f"d_loss_{key}"] = loss
    
    return losses


def compute_generator_losses(
    disc_outputs_fake: dict[str, tuple[list[torch.Tensor], list[list[torch.Tensor]]]],
    disc_outputs_real: dict[str, tuple[list[torch.Tensor], list[list[torch.Tensor]]]],
    fm_weight: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Compute generator adversarial and feature matching losses."""
    losses = {}
    
    for key in disc_outputs_fake:
        fake_outs, fake_feats = disc_outputs_fake[key]
        real_outs, real_feats = disc_outputs_real[key]
        
        adv_loss = generator_loss(fake_outs)
        fm_loss = feature_matching_loss(real_feats, fake_feats)
        
        losses[f"g_adv_{key}"] = adv_loss
        losses[f"g_fm_{key}"] = fm_loss
    
    return losses
