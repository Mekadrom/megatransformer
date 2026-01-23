from dataclasses import dataclass
import dataclasses
import json


@dataclass
class VocoderConfig:
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
    "tiny": VocoderConfig(),
}
