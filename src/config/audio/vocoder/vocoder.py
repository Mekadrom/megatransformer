from dataclasses import dataclass
import dataclasses
import json


@dataclass
class VocoderConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    hidden_dim: int = 512
    num_conv_layers: int = 6
    num_attn_layers: int = 2
    attn_heads: int = 4
    convnext_mult: int = 4
    attn_dropout: float = 0.0
    cutoff_bin: int = 128
    low_freq_kernel: int = 7
    high_freq_kernel: int = 3
    sc_loss_weight: float = 1.0
    mag_loss_weight: float = 3.0
    waveform_l1_loss_weight: float = 0.1
    mel_recon_loss_weight: float = 1.0
    mel_recon_loss_weight_linspace_max: float = 1.0
    complex_stft_loss_weight: float = 2.0
    phase_loss_weight: float = 1.0
    phase_ip_loss_weight: float = 1.0
    phase_iaf_loss_weight: float = 1.0
    phase_gd_loss_weight: float = 1.0
    high_freq_stft_loss_weight: float = 0.0
    high_freq_stft_cutoff_bin: int = 256
    direct_mag_loss_weight: float = 0.0
    wav2vec2_loss_weight: float = 0.0
    wav2vec2_model: str = "facebook/wav2vec2-base"


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)
    
    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


VOCODER_CONFIGS = {
    "default": VocoderConfig(),
    "tiny": VocoderConfig(hidden_dim=128),
}
