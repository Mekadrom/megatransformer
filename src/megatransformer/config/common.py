import dataclasses
import json


from dataclasses import dataclass
from typing import Optional


@dataclass
class MegaTransformerBlockConfig:
    d_model: int = 512
    n_heads: int = 6
    d_queries: int = 64
    d_values: int = 64
    n_query_groups: int = 1
    d_inner: int = 2048
    n_layers: int = 6
    heads_activation: Optional[str] = None
    use_qkv_bias: bool = True
    use_rotary_embedding: bool = False
    rotary_embedding_dim: int = 64
    rotary_embedding_learnable: bool = False
    use_grok_scaled_attn: bool = False
    use_alibi_bias: bool = False
    max_position_embeddings: int = 1024
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    ffn_type: str = "mlp"
    activation_function: str = "gelu"
    causal: bool = True  # If False, no causal mask (bidirectional attention)

    # Soft logit capping on attention scores (Gemma 2-style). Applied after
    # scaling by 1/sqrt(d_queries): scores = cap * tanh(scores / cap).
    # Prevents attention entropy collapse without hard clipping.
    # None = disabled (default). Typical values: 30.0–50.0.
    attn_logit_cap: Optional[float] = None

    norm_type: str = "layernorm"
    norm_eps: float = 1e-5
    pre_attn_norm: bool = True
    inter_attn_norm: bool = False  # only applies to blocks with cross-attention
    post_attn_norm: bool = False
    pre_ffn_norm: bool = True
    post_ffn_norm: bool = False

    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class AudioConfig:
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    sample_rate: int = 16000
    max_audio_duration: float = 30.0  # in seconds
    latent_channels: int = 8
    latent_compression_factor: tuple[int, int] = (8, 12)


@dataclass
class ImageConfig:
    image_size: int = 256
    latent_channels: int = 12
    latent_compression_factor: int = 8
    latent_patch_size: int = 4

