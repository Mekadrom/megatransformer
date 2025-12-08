from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

import megatransformer_utils
from model.activations import Snake, SwiGLU
from model.audio.shared_window_buffer import SharedWindowBuffer


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block - efficient and effective for audio.
    Depthwise conv -> pointwise expand -> pointwise contract
    """
    def __init__(self, dim, ovr_out_dim=None, kernel_size=7, expansion=4):
        super().__init__()
        self.ovr_out_dim = ovr_out_dim

         # depthwise conv1d
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
        
        # pointwise (linear is good enough)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, ovr_out_dim if ovr_out_dim is not None else dim)
    
    def forward(self, x):
        # x: [B, C, T]
        residual = x
        
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, T, C] for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # [B, C, T]
        
        if self.ovr_out_dim is not None:
            return x  # no residual if changing channels
        return residual + x


class FrequencyDomainVocoderBase(nn.Module):
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = n_fft // 2 + 1  # 513 for n_fft=1024
        
        # input projection: mel bins -> hidden
        self.input_proj = nn.Conv1d(n_mels, hidden_dim, kernel_size=7, padding=3)
        
        # iSTFT window
        self.register_buffer('window', shared_window_buffer.get_window(n_fft, torch.device('cpu')))
        

    def _init_weights(self):
        # Input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        
        # Magnitude head - small init so initial output is small positive
        for m in self.mag_head:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Phase head - small init for stable start
        for m in self.phase_head:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, mel):
        """
        Args:
            mel: [B, n_mels, T] log mel spectrogram

        Returns:
            waveform: [B, T * hop_length]
            stft: [B, freq_bins, T] complex - for loss computation
        """
        # Backbone
        x = self.input_proj(mel)

        for block in self.backbone:
            x = block(x)  # Residual is applied in the block forward

        # Predict magnitude via ELU + 1: outputs in (0, ∞)
        # ELU + 1 has softplus-like bounded gradients but no floor artifact
        # At x=0: output=1, at x=-∞: output→0, at x=+∞: linear growth
        mag_pre = self.mag_head(x)
        mag = F.elu(mag_pre, alpha=1.0) + 1.0

        # Predict phase angle directly: [B, 513, T]
        # Then use cos/sin to get unit phasor - always valid, no normalization needed
        phase_angle = self.phase_head(x)

        # Convert angle to unit phasor via cos/sin - guaranteed to be on unit circle
        phase_real = torch.cos(phase_angle)
        phase_imag = torch.sin(phase_angle)

        # Construct complex STFT: magnitude * e^(i*phase)
        stft_real = mag * phase_real
        stft_imag = mag * phase_imag
        stft = torch.complex(stft_real.to(torch.float32), stft_imag.to(torch.float32))

        # iSTFT to waveform
        waveform = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(stft.device),
            length=mel.size(-1) * self.hop_length,
            return_complex=False,
        )

        return waveform, stft
    
    def get_phase(self, stft):
        """Extract phase angle from predicted STFT for loss computation."""
        return torch.angle(stft)
    
    def get_magnitude(self, stft):
        """Extract magnitude from predicted STFT for loss computation."""
        return stft.abs()


class HeavyHeadedFrequencyDomainVocoder(FrequencyDomainVocoderBase):
    """
    Predicts complex STFT (magnitude + phase) from mel spectrogram,
    then uses iSTFT for lossless waveform reconstruction.

    No upsampling needed - mel frames and STFT frames have same time resolution.
    """
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
        convnext_mult: int = 4,
        phase_activation_fn: str = 'sigmoid',
    ):
        super().__init__(shared_window_buffer, n_mels, n_fft, hop_length, hidden_dim)

        # backbone - no upsampling, just channel transformations
        self.backbone = nn.ModuleList([
            ConvNeXtBlock(hidden_dim, expansion=convnext_mult) for _ in range(num_layers)
        ])

        # magnitude head - outputs log-magnitude, exp() applied in forward
        self.mag_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, self.freq_bins, kernel_size=7, padding=3),
        )

        act_type = megatransformer_utils.get_activation_type(phase_activation_fn)
        self.act: nn.Module = act_type(hidden_dim) if act_type in [Snake, SwiGLU] else act_type()

        # Phase head - outputs angle directly, then we use cos/sin for unit phasor
        # Snake activation is periodic-aware, might be good for angle prediction
        self.phase_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            self.act,
            nn.Conv1d(hidden_dim, self.freq_bins, kernel_size=7, padding=3),  # just one output per bin (the angle)
        )

        self._init_weights()


class LightHeadedFrequencyDomainVocoder(FrequencyDomainVocoderBase):
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
        convnext_mult: int = 4,
    ):
        super().__init__(shared_window_buffer, n_mels, n_fft, hop_length, hidden_dim)

        # backbone - no upsampling, just channel transformations
        backbone_layers = [
            ConvNeXtBlock(hidden_dim, expansion=convnext_mult) for _ in range(num_layers - 1)
        ]
        backbone_layers.append(ConvNeXtBlock(hidden_dim, ovr_out_dim=hidden_dim // 2, expansion=convnext_mult))

        self.backbone = nn.ModuleList(backbone_layers)

        # magnitude head - outputs log-magnitude, exp() applied in forward
        self.mag_head = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, self.freq_bins, kernel_size=7, padding=3),
        )

        self.phase_head = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, self.freq_bins, kernel_size=7, padding=3),
        )

        self._init_weights()


class SplitBandFrequencyDomainVocoder(FrequencyDomainVocoderBase):
    """
    Frequency domain vocoder with separate heads for low and high frequency bands.

    Key differences from LightHeaded:
    - Low-freq head: larger kernels for better frequency resolution, SiLU activation
    - High-freq head: smaller kernels for temporal precision, Snake activation (periodic bias)
    - Separate phase heads per band with same kernel size philosophy

    Parameter-matched to LightHeaded by using single-layer heads (no intermediate expansion).
    """
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
        convnext_mult: int = 4,
        cutoff_bin: int = 128,  # ~2kHz at 16kHz sample rate with n_fft=1024
        low_freq_kernel: int = 7,
        high_freq_kernel: int = 3,
    ):
        super().__init__(shared_window_buffer, n_mels, n_fft, hop_length, hidden_dim)

        self.cutoff_bin = cutoff_bin
        self.n_low_bins = cutoff_bin
        self.n_high_bins = self.freq_bins - cutoff_bin

        # Backbone - same as LightHeaded
        backbone_layers = [
            ConvNeXtBlock(hidden_dim, expansion=convnext_mult) for _ in range(num_layers - 1)
        ]
        backbone_layers.append(ConvNeXtBlock(hidden_dim, ovr_out_dim=hidden_dim // 2, expansion=convnext_mult))
        self.backbone = nn.ModuleList(backbone_layers)

        head_input_dim = hidden_dim // 2

        # ============ Magnitude Heads ============
        # Single-layer heads to match LightHeaded parameter count
        # Low-freq: larger kernel for frequency resolution
        self.mag_head_low = nn.Conv1d(
            head_input_dim, self.n_low_bins,
            kernel_size=low_freq_kernel, padding=low_freq_kernel // 2
        )

        # High-freq: smaller kernel for temporal precision
        self.mag_head_high = nn.Conv1d(
            head_input_dim, self.n_high_bins,
            kernel_size=high_freq_kernel, padding=high_freq_kernel // 2
        )

        # ============ Phase Heads ============
        # Low-freq: larger kernel for phase coherence, SiLU for smooth output
        self.phase_head_low = nn.Sequential(
            nn.Conv1d(head_input_dim, self.n_low_bins, kernel_size=low_freq_kernel, padding=low_freq_kernel // 2),
            nn.SiLU(),
        )

        # High-freq: smaller kernel, Snake for periodic inductive bias
        self.phase_head_high_snake = Snake(head_input_dim)
        self.phase_head_high_conv = nn.Conv1d(
            head_input_dim, self.n_high_bins,
            kernel_size=high_freq_kernel, padding=high_freq_kernel // 2
        )

        self._init_weights()

    def _init_weights(self):
        # Input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        # Magnitude heads - small init for stable start
        for head in [self.mag_head_low, self.mag_head_high]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

        # Phase heads - small init
        # Low-freq phase head (Sequential with conv)
        for m in self.phase_head_low:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # High-freq phase head conv
        nn.init.xavier_uniform_(self.phase_head_high_conv.weight, gain=0.1)
        if self.phase_head_high_conv.bias is not None:
            nn.init.zeros_(self.phase_head_high_conv.bias)

    def forward(self, mel):
        """
        Args:
            mel: [B, n_mels, T] log mel spectrogram

        Returns:
            waveform: [B, T * hop_length]
            stft: [B, freq_bins, T] complex - for loss computation
        """
        # Backbone
        x = self.input_proj(mel)
        for block in self.backbone:
            x = block(x)

        # ============ Magnitude prediction via ELU + 1 ============
        # ELU + 1 has softplus-like bounded gradients but no floor artifact
        mag_pre_low = self.mag_head_low(x)    # [B, n_low_bins, T]
        mag_pre_high = self.mag_head_high(x)  # [B, n_high_bins, T]

        # Concatenate bands and apply ELU + 1
        mag_pre = torch.cat([mag_pre_low, mag_pre_high], dim=1)  # [B, freq_bins, T]
        mag = F.elu(mag_pre, alpha=1.0) + 1.0

        # ============ Phase prediction ============
        phase_low = self.phase_head_low(x)    # [B, n_low_bins, T]
        # High-freq phase: Snake activation then conv
        phase_high = self.phase_head_high_conv(self.phase_head_high_snake(x))  # [B, n_high_bins, T]

        # Concatenate bands
        phase_angle = torch.cat([phase_low, phase_high], dim=1)  # [B, freq_bins, T]

        # Convert angle to unit phasor
        phase_real = torch.cos(phase_angle)
        phase_imag = torch.sin(phase_angle)

        # Construct complex STFT
        stft_real = mag * phase_real
        stft_imag = mag * phase_imag
        stft = torch.complex(stft_real.to(torch.float32), stft_imag.to(torch.float32))

        # iSTFT to waveform
        waveform = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(stft.device),
            length=mel.size(-1) * self.hop_length,
            return_complex=False,
        )

        return waveform, stft
