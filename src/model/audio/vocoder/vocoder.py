from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.audio.vocoder.vocoder import VOCODER_CONFIGS, VocoderConfig
from model import activations
from model.audio.vocoder.convnext import ConvNeXtBlock
from model.audio.vocoder.frequency_attention_block import FrequencyAttentionBlock
from utils.audio_utils import SharedWindowBuffer


class Vocoder(nn.Module):
    """
    Frequency domain vocoder with attention for improved phase coherence.

    Interleaves ConvNeXt blocks with attention blocks to capture:
    - Local patterns via convolution
    - Global phase relationships via attention

    Attention helps harmonics (which are related across time) maintain coherence.
    """
    def __init__(self, shared_window_buffer: SharedWindowBuffer, config: VocoderConfig):
        super().__init__()

        self.config = config

        self.freq_bins = config.n_fft // 2 + 1  # 513 for n_fft=1024
        
        # input projection: mel bins -> hidden
        self.input_proj = nn.Conv1d(config.n_mels, config.hidden_dim, kernel_size=7, padding=3)

        # iSTFT window
        self.register_buffer('window', shared_window_buffer.get_window(config.n_fft, torch.device('cpu')))

        self.cutoff_bin = config.cutoff_bin
        self.n_low_bins = config.cutoff_bin
        self.n_high_bins = self.freq_bins - config.cutoff_bin

        # Build backbone: ConvNeXt blocks with attention interspersed
        # Pattern: [Conv, Conv, ..., Attn, Conv, Conv, ..., Attn, ...]
        backbone_layers = []
        conv_per_attn = config.num_conv_layers // (config.num_attn_layers + 1)

        for i in range(config.num_attn_layers + 1):
            # Add ConvNeXt blocks
            for _ in range(conv_per_attn):
                backbone_layers.append(ConvNeXtBlock(config.hidden_dim, expansion=config.convnext_mult))

            # Add attention after each group (except the last)
            if i < config.num_attn_layers:
                backbone_layers.append(
                    FrequencyAttentionBlock(
                        dim=config.hidden_dim,
                        num_heads=config.attn_heads,
                        dropout=config.attn_dropout,
                    )
                )

        # Final ConvNeXt to reduce channels
        backbone_layers.append(
            ConvNeXtBlock(config.hidden_dim, ovr_out_dim=config.hidden_dim // 2, expansion=config.convnext_mult)
        )

        self.backbone = nn.ModuleList(backbone_layers)

        head_input_dim = config.hidden_dim // 2

        # Split-band magnitude heads
        self.mag_head_low = nn.Conv1d(
            head_input_dim, self.n_low_bins,
            kernel_size=config.low_freq_kernel, padding=config.low_freq_kernel // 2
        )
        self.mag_head_high = nn.Conv1d(
            head_input_dim, self.n_high_bins,
            kernel_size=config.high_freq_kernel, padding=config.high_freq_kernel // 2
        )

        # Phase heads with attention-informed features
        self.phase_head_low = nn.Sequential(
            nn.Conv1d(head_input_dim, head_input_dim, kernel_size=config.low_freq_kernel, padding=config.low_freq_kernel // 2),
            activations.Snake(head_input_dim),
            nn.Conv1d(head_input_dim, self.n_low_bins, kernel_size=config.low_freq_kernel, padding=config.low_freq_kernel // 2),
        )

        self.phase_head_high = nn.Sequential(
            activations.Snake(head_input_dim),
            nn.Conv1d(head_input_dim, self.n_high_bins, kernel_size=config.high_freq_kernel, padding=config.high_freq_kernel // 2)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        for head in [self.mag_head_low, self.mag_head_high]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

        for m in self.phase_head_low:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.phase_head_high:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @classmethod
    def from_config(cls, config_name: str, shared_window_buffer: Optional[SharedWindowBuffer], **overrides) -> "Vocoder":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = Vocoder.from_config("tiny", hidden_dim=256)
        """
        if config_name not in VOCODER_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(VOCODER_CONFIGS.keys())}")

        config = VOCODER_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = VocoderConfig(**config_dict)

        if shared_window_buffer is None:
            shared_window_buffer = SharedWindowBuffer()

        return cls(shared_window_buffer, config)

    def get_phase(self, stft: torch.Tensor) -> torch.Tensor:
        """Extract phase angle from predicted STFT for loss computation."""
        return torch.angle(stft)
    
    def get_magnitude(self, stft: torch.Tensor) -> torch.Tensor:
        """Extract magnitude from predicted STFT for loss computation."""
        return stft.abs()

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: [B, n_mels, T] log mel spectrogram
            OR [B, 1, n_mels, T] with singleton channel dim (gets dropped)

        Returns:
            waveform: [B, T * hop_length]
            stft: [B, freq_bins, T] complex - for loss computation
        """
        if mel.dim() == 4 and mel.size(1) == 1:
            mel = mel.squeeze(1)  # Remove singleton channel dim if present

        x = self.input_proj(mel)

        for block in self.backbone:
            x = block(x)

        # Magnitude prediction
        mag_pre_low = self.mag_head_low(x)
        mag_pre_high = self.mag_head_high(x)
        mag_pre = torch.cat([mag_pre_low, mag_pre_high], dim=1)
        mag = F.elu(mag_pre, alpha=1.0) + 1.0

        # Phase prediction
        phase_low = self.phase_head_low(x)
        phase_high = self.phase_head_high(x)
        phase_angle = torch.cat([phase_low, phase_high], dim=1)

        phase_real = torch.cos(phase_angle)
        phase_imag = torch.sin(phase_angle)

        # Construct complex STFT
        stft_real = mag * phase_real
        stft_imag = mag * phase_imag
        stft = torch.complex(stft_real.to(torch.float32), stft_imag.to(torch.float32))

        # iSTFT to waveform
        waveform = torch.istft(
            stft,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window=self.window.to(stft.device),
            length=mel.size(-1) * self.config.hop_length,
            return_complex=False,
        )

        return waveform, stft
