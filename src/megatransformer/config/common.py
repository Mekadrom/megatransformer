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
    # M-RoPE: split the rotary dims into a GLOBAL half (strictly increasing across the whole
    # interleaved sequence — disambiguates voice_0 from voice_1 in multi-example sequences)
    # and a LOCAL half (index within the current same-modality segment, with voice scaled by
    # 1/mrope_voice_rate). The local axis puts an aligned (text token j, voice frame t) pair at
    # relative distance ~0, which is what RoPE's locality bias can actually exploit — today the
    # text->voice offset is L_text + 0.83*t, i.e. large, growing, and utterance-dependent.
    # Same rotary mechanism, different position integers.
    use_mrope: bool = False
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

    # MuonClip / qk-clip (Kimi K2). NOT the same thing as attn_logit_cap above: that one
    # softcaps scores in the FORWARD pass (and costs the SDPA fast path, transformer.py:271);
    # this one leaves the forward pass alone and rescales the W_q / W_k WEIGHTS after an
    # optimizer step whenever an observed per-head max logit exceeds qk_clip_tau. It exists
    # because Muon's orthogonalized updates give every singular direction the same step size,
    # which inflates spectral norms and can run attention logits away over a long run.
    #
    # None = disabled, and disabled is BIT-IDENTICAL: the forward-pass probe is gated on a
    # runtime flag the controller only arms every qk_clip_probe_every steps, and it runs
    # under no_grad writing to a non-parameter attribute.
    qk_clip_tau: Optional[float] = None

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

