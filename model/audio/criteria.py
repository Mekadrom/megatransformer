import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from typing import Optional

from speechbrain.inference.speaker import EncoderClassifier

from utils import configuration
from utils.audio_utils import configurable_mel_spectrogram, SharedWindowBuffer


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
            win_length=self.n_fft,
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


class PhaseLoss(nn.Module):
    def __init__(self, shared_window_buffer: SharedWindowBuffer, config: configuration.MegaTransformerConfig):
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

        log_target_hf = torch.log(target_hf.clamp(min=1e-7))
        log_pred_hf = torch.log(pred_hf.clamp(min=1e-7))

        # Weight by target magnitude - emphasize bins with actual signal
        # Normalize per-frame so weights sum to 1 across frequency bins
        weights = target_hf / (target_hf.sum(dim=-2, keepdim=True) + 1e-5)

        weighted_loss = weights * (log_pred_hf - log_target_hf).abs()
        return weighted_loss.sum(dim=-2).mean()  # sum over freq, mean over batch/time


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


# =============================================================================
# Audio Perceptual Losses
# =============================================================================


class Wav2Vec2PerceptualLoss(nn.Module):
    """
    Perceptual loss using Wav2Vec2 features.

    Wav2Vec2 is a self-supervised speech representation model that learns
    rich acoustic features. Using it for perceptual loss helps the VAE
    learn phonetically meaningful reconstructions.

    Model sizes:
        - 'facebook/wav2vec2-base': ~95M parameters
        - 'facebook/wav2vec2-large': ~317M parameters (better but slower)
        - 'facebook/wav2vec2-large-960h': ~317M parameters (fine-tuned on LibriSpeech)

    Requires: pip install transformers
    """
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        feature_layers: list[int] = None,
        sample_rate: int = 16000,
    ):
        super().__init__()
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")

        self.sample_rate = sample_rate
        self.model_name = model_name

        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze all parameters
        self.model.requires_grad_(False)
        self.model.eval()

        # Which transformer layers to extract features from
        # Default: use layers from different depths for multi-scale features
        if feature_layers is None:
            n_layers = self.model.config.num_hidden_layers
            # Sample layers evenly: early, middle, late
            feature_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        self.feature_layers = feature_layers

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def extract_features(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract features from multiple transformer layers.

        Args:
            waveform: [B, T] waveform at self.sample_rate

        Returns:
            List of feature tensors from specified layers, each [B, T', hidden_size]
        """
        # Ensure float32 for wav2vec2
        waveform = waveform.float()

        # Forward pass with hidden states output
        outputs = self.model(
            waveform,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract features from specified layers
        hidden_states = outputs.hidden_states  # Tuple of [B, T', hidden_size]
        features = [hidden_states[i] for i in self.feature_layers]

        return features

    def forward(self, pred_waveform: torch.Tensor, target_waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute Wav2Vec2 perceptual loss.

        Args:
            pred_waveform: [B, T] or [B, 1, T] predicted waveform
            target_waveform: [B, T] or [B, 1, T] target waveform

        Returns:
            Scalar loss value
        """
        # Handle [B, 1, T] input
        if pred_waveform.dim() == 3:
            pred_waveform = pred_waveform.squeeze(1)
        if target_waveform.dim() == 3:
            target_waveform = target_waveform.squeeze(1)

        # Match lengths
        min_len = min(pred_waveform.shape[-1], target_waveform.shape[-1])
        pred_waveform = pred_waveform[..., :min_len]
        target_waveform = target_waveform[..., :min_len]

        # Extract features
        with torch.no_grad():
            target_features = self.extract_features(target_waveform)

        pred_features = self.extract_features(pred_waveform)

        # Compute L1 loss across all layers
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            # Match sequence lengths (can differ slightly due to conv layers)
            min_t = min(pred_feat.shape[1], target_feat.shape[1])
            loss = loss + F.l1_loss(
                pred_feat[:, :min_t],
                target_feat[:, :min_t].detach()
            )

        return loss / len(self.feature_layers)


class PANNsPerceptualLoss(nn.Module):
    """
    Perceptual loss using PANNs (Pretrained Audio Neural Networks) features.

    PANNs are CNNs trained on AudioSet for audio tagging. They learn
    general audio features that are useful for perceptual similarity.

    Model: CNN14 (~80M parameters)

    Requires: pip install panns-inference
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        model_type: str = "Cnn14",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_sample_rate = 32000  # PANNs uses 32kHz

        try:
            from panns_inference import AudioTagging
        except ImportError:
            raise ImportError("panns-inference not installed. Run: pip install panns-inference")

        # Load PANNs model
        self.model = AudioTagging(checkpoint_path=None, device='cpu')
        self.cnn = self.model.model

        # Freeze all parameters
        self.cnn.requires_grad_(False)
        self.cnn.eval()

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
        else:
            self.resampler = None

    def train(self, mode=True):
        super().train(mode)
        self.cnn.eval()
        return self

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features from PANNs.

        Args:
            waveform: [B, T] waveform

        Returns:
            Feature tensor [B, feature_dim]
        """
        # Ensure float32
        waveform = waveform.float()

        # Resample to 32kHz if needed
        if self.resampler is not None:
            waveform = self.resampler(waveform)

        # Move model to same device
        if next(self.cnn.parameters()).device != waveform.device:
            self.cnn.to(waveform.device)
            if self.resampler is not None:
                self.resampler.to(waveform.device)

        # Forward through CNN to get embedding (before final classification)
        # PANNs CNN14 returns dict with 'embedding' (2048-dim) and 'clipwise_output'
        output_dict = self.cnn(waveform)
        return output_dict['embedding']

    def forward(self, pred_waveform: torch.Tensor, target_waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute PANNs perceptual loss.

        Args:
            pred_waveform: [B, T] or [B, 1, T] predicted waveform
            target_waveform: [B, T] or [B, 1, T] target waveform

        Returns:
            Scalar loss value
        """
        # Handle [B, 1, T] input
        if pred_waveform.dim() == 3:
            pred_waveform = pred_waveform.squeeze(1)
        if target_waveform.dim() == 3:
            target_waveform = target_waveform.squeeze(1)

        # Match lengths
        min_len = min(pred_waveform.shape[-1], target_waveform.shape[-1])
        pred_waveform = pred_waveform[..., :min_len]
        target_waveform = target_waveform[..., :min_len]

        # Extract features
        with torch.no_grad():
            target_features = self.extract_features(target_waveform)

        pred_features = self.extract_features(pred_waveform)

        # L1 loss on embeddings
        return F.l1_loss(pred_features, target_features.detach())


class MultiScaleMelSpectrogramLoss(nn.Module):
    """
    Multi-scale mel spectrogram loss for mel-to-mel comparison.

    Unlike MultiScaleMelLoss which operates on waveforms, this loss
    compares mel spectrograms directly at multiple resolutions by
    applying different smoothing/pooling operations.

    This is useful for VAE training where we have mel spectrograms
    directly without needing a vocoder.
    """
    def __init__(
        self,
        scales: list[int] = None,
        use_log: bool = True,
    ):
        super().__init__()

        # Different pooling scales for multi-resolution comparison
        if scales is None:
            scales = [1, 2, 4, 8]  # No pooling, 2x, 4x, 8x downsampling

        self.scales = scales
        self.use_log = use_log

        # Create average pooling layers for each scale > 1
        self.pooling_layers = nn.ModuleDict()
        for scale in scales:
            if scale > 1:
                self.pooling_layers[str(scale)] = nn.AvgPool2d(
                    kernel_size=(1, scale),  # Only pool in time dimension
                    stride=(1, scale),
                )

    def forward(self, pred_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale mel loss.

        Args:
            pred_mel: [B, 1, n_mels, T] or [B, n_mels, T] predicted mel
            target_mel: [B, 1, n_mels, T] or [B, n_mels, T] target mel

        Returns:
            Scalar loss value
        """
        # Ensure 4D tensors [B, 1, n_mels, T]
        if pred_mel.dim() == 3:
            pred_mel = pred_mel.unsqueeze(1)
        if target_mel.dim() == 3:
            target_mel = target_mel.unsqueeze(1)

        # Match lengths
        min_len = min(pred_mel.shape[-1], target_mel.shape[-1])
        pred_mel = pred_mel[..., :min_len]
        target_mel = target_mel[..., :min_len]

        total_loss = 0.0

        for scale in self.scales:
            if scale == 1:
                pred_scaled = pred_mel
                target_scaled = target_mel
            else:
                pool = self.pooling_layers[str(scale)]
                pred_scaled = pool(pred_mel)
                target_scaled = pool(target_mel)

            # Apply log if using log mel
            if self.use_log:
                pred_log = torch.log(pred_scaled.clamp(min=1e-5))
                target_log = torch.log(target_scaled.clamp(min=1e-5))
                total_loss = total_loss + F.l1_loss(pred_log, target_log)
            else:
                total_loss = total_loss + F.l1_loss(pred_scaled, target_scaled)

        return total_loss / len(self.scales)


class AudioPerceptualLoss(nn.Module):
    """
    Combined audio perceptual loss with configurable components.

    Combines multiple audio perceptual losses with individual weights:
    - Multi-scale mel loss (mel-to-mel comparison)
    - Wav2Vec2 features (speech-specific)
    - PANNs features (general audio)

    Note: Wav2Vec2 and PANNs losses require waveforms, so a vocoder
    must be used to convert mel spectrograms to waveforms first.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        # Component weights (0 = disabled)
        multi_scale_mel_weight: float = 1.0,
        wav2vec2_weight: float = 1.0,
        panns_weight: float = 0.0,  # Disabled by default (requires extra dependency)
        speaker_embedding_weight: float = 0.0,
        # Wav2Vec2 settings
        wav2vec2_model: str = "facebook/wav2vec2-base",
        wav2vec2_layers: list[int] = None,
        # Multi-scale mel settings
        mel_scales: list[int] = None,
        use_log_mel: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.multi_scale_mel_weight = multi_scale_mel_weight
        self.wav2vec2_weight = wav2vec2_weight
        self.panns_weight = panns_weight
        self.speaker_embedding_weight = speaker_embedding_weight

        # Initialize enabled losses
        self.multi_scale_mel_loss = None
        self.wav2vec2_loss = None
        self.panns_loss = None
        self.speaker_embedding_loss = None

        if multi_scale_mel_weight > 0:
            self.multi_scale_mel_loss = MultiScaleMelSpectrogramLoss(
                scales=mel_scales,
                use_log=use_log_mel,
            )

        if wav2vec2_weight > 0:
            self.wav2vec2_loss = Wav2Vec2PerceptualLoss(
                model_name=wav2vec2_model,
                feature_layers=wav2vec2_layers,
                sample_rate=sample_rate,
            )

        if panns_weight > 0:
            self.panns_loss = PANNsPerceptualLoss(
                sample_rate=sample_rate,
            )

        if speaker_embedding_weight > 0:
            self.speaker_embedding_loss = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb-mel-spec",
                savedir="pretrained_models/spkrec-ecapa-voxceleb-mel-spec",
            )

    def forward(
        self,
        pred_mel: torch.Tensor = None,
        target_mel: torch.Tensor = None,
        target_speaker_embedding: torch.Tensor = None,
        pred_waveform: torch.Tensor = None,
        target_waveform: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute audio perceptual losses.

        Args:
            pred_mel: [B, 1, n_mels, T] predicted mel spectrogram
            target_mel: [B, 1, n_mels, T] target mel spectrogram
            pred_waveform: [B, T] predicted waveform (for wav2vec2/panns)
            target_waveform: [B, T] target waveform (for wav2vec2/panns)

        Returns:
            Dict with individual losses and total perceptual loss
        """
        device = pred_mel.device if pred_mel is not None else pred_waveform.device
        losses = {}
        total = torch.tensor(0.0, device=device)

        # Multi-scale mel loss (works on mel spectrograms)
        if self.multi_scale_mel_loss is not None and pred_mel is not None:
            ms_mel_loss = self.multi_scale_mel_loss(pred_mel, target_mel)
            losses["multi_scale_mel_loss"] = ms_mel_loss
            total = total + self.multi_scale_mel_weight * ms_mel_loss
        else:
            losses["multi_scale_mel_loss"] = torch.tensor(0.0, device=device)

        # Wav2Vec2 loss (requires waveforms)
        if self.wav2vec2_loss is not None and pred_waveform is not None:
            w2v_loss = self.wav2vec2_loss(pred_waveform, target_waveform)
            losses["wav2vec2_loss"] = w2v_loss
            total = total + self.wav2vec2_weight * w2v_loss
        else:
            losses["wav2vec2_loss"] = torch.tensor(0.0, device=device)

        # PANNs loss (requires waveforms)
        if self.panns_loss is not None and pred_waveform is not None:
            panns_loss = self.panns_loss(pred_waveform, target_waveform)
            losses["panns_loss"] = panns_loss
            total = total + self.panns_weight * panns_loss
        else:
            losses["panns_loss"] = torch.tensor(0.0, device=device)

        # Speaker embedding loss (requires waveforms)
        if self.speaker_embedding_loss is not None:
            # Extract embeddings
            pred_emb = self.speaker_embedding_loss.encode_batch(pred_waveform)
            spk_loss = F.l1_loss(pred_emb, target_speaker_embedding)
            losses["speaker_embedding_loss"] = spk_loss
            total = total + self.speaker_embedding_weight * spk_loss
            

        losses["total_perceptual_loss"] = total
        return losses
