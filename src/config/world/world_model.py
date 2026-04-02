import dataclasses
import json

from dataclasses import dataclass
from typing import Optional, List

from config.audio.feature_extractor import AudioVAEPreludeFeatureExtractorConfig
from config.audio.generator import AudioCodaAndVAEConfig
from config.common import MegaTransformerBlockConfig
from config.image.feature_extractor import ImageVAEPreludeFeatureExtractorConfig
from config.image.generator import ImageCodaAndVAEConfig
from config.text.feature_extractor import TextPreludeFeatureExtractorConfig
from config.text.generator import TextCodaClassifierConfig
from utils.constants import (
    AUDIO_PLACEHOLDER_TOKEN_ID,
    VOICE_PLACEHOLDER_TOKEN_ID,
    IMAGE_PLACEHOLDER_TOKEN_ID,
)


@dataclass
class TokenInterleaverConfig:
    """Configuration for TokenInterleaver with placeholder token IDs.

    These token IDs should match your tokenizer's vocabulary for the special
    placeholder tokens that mark where media examples should be inserted.

    For example, if your tokenizer has:
        - <audio> at token ID 32000
        - <voice> at token ID 32001
        - <image> at token ID 32002

    Then configure accordingly. The interleaver will scan for these tokens
    in the input sequence and replace them with the corresponding media embeddings.
    """
    audio_placeholder_token_id: Optional[int] = AUDIO_PLACEHOLDER_TOKEN_ID
    voice_placeholder_token_id: Optional[int] = VOICE_PLACEHOLDER_TOKEN_ID
    image_placeholder_token_id: Optional[int] = IMAGE_PLACEHOLDER_TOKEN_ID


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MegaTransformerRecurrentConfig:
    """Configuration for the recurrent (thought vector) block.

    The recurrent block operates at 2*d_model internally to concatenate
    input embeddings with thought state, then projects back to d_model.
    """
    block_config: MegaTransformerBlockConfig = dataclasses.field(
        default_factory=lambda: MegaTransformerBlockConfig(d_model=1024, d_inner=4096)
    )
    n_recurrent_blocks: int = 1  # Number of distinct blocks cycled per iteration (Huginn uses 4)
    injection_type: str = "concat"  # "concat" (2h blocks + projection) or "add" (additive, Huginn-style)
    mean_thinking_steps: int = 32
    backprop_depth: int = 8
    thought_initialization_method: str = "like-init"
    thought_init_std: float = 0.02  # Huginn uses sqrt(2/5) ≈ 0.6325
    depth_scaled_init: bool = True  # Scale output projection init by 1/sqrt(5*h*l_eff)
    projection_init_gain: float = 1.0  # Gain multiplier for the thought projection init
    block_init_gain: float = 0.02  # Xavier gain for recurrent block weights (default matches transformer_weight_init)
    iteration_norm: str = "none"  # "none", "pre_projection", or "post_projection"
    share_block_weights: bool = False  # If True, all recurrent blocks share weights (deeper 1-block)
    exit_criteria: str = "kl_divergence"
    exit_criteria_threshold: float = 1e-4
    lockstep_n: bool = False
    lockstep_k: bool = False

    
    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MegaTransformerWorldModelConfig:
    """
    Configuration for the Megatransformer world model, integrating audio, image, and text modalities.

    include_modes controls which modality-specific preludes and codas are instantiated.
    Text is always included. Omitted modalities save memory and parameters.
    """
    # Which modalities to instantiate (text is always included)
    include_modes: List[str] = dataclasses.field(
        default_factory=lambda: ["text", "audio", "voice", "image"]
    )

    # Feature extractor configs
    text_prelude_config: TextPreludeFeatureExtractorConfig = dataclasses.field(
        default_factory=TextPreludeFeatureExtractorConfig
    )
    audio_prelude_config: AudioVAEPreludeFeatureExtractorConfig = dataclasses.field(
        default_factory=AudioVAEPreludeFeatureExtractorConfig
    )
    voice_prelude_config: AudioVAEPreludeFeatureExtractorConfig = dataclasses.field(
        default_factory=AudioVAEPreludeFeatureExtractorConfig
    )
    image_prelude_config: ImageVAEPreludeFeatureExtractorConfig = dataclasses.field(
        default_factory=ImageVAEPreludeFeatureExtractorConfig
    )
    # Token interleaver config
    token_interleaver_config: TokenInterleaverConfig = dataclasses.field(
        default_factory=TokenInterleaverConfig
    )
    # Main transformer config
    recurrent_block_config: MegaTransformerRecurrentConfig = dataclasses.field(
        default_factory=MegaTransformerRecurrentConfig
    )
    # Coda/generator configs
    text_coda_config: TextCodaClassifierConfig = dataclasses.field(
        default_factory=TextCodaClassifierConfig
    )
    audio_coda_config: AudioCodaAndVAEConfig = dataclasses.field(
        default_factory=AudioCodaAndVAEConfig
    )
    voice_coda_config: AudioCodaAndVAEConfig = dataclasses.field(
        default_factory=AudioCodaAndVAEConfig
    )
    image_coda_config: ImageCodaAndVAEConfig = dataclasses.field(
        default_factory=ImageCodaAndVAEConfig
    )

    # Cross-attention image decoder (optional, for non-autoregressive image generation)
    image_cross_decoder_config: Optional[dict] = None  # None = disabled

    # Scale embeddings by sqrt(d_model) before recurrent block (Huginn-style)
    scale_embeddings: bool = False

    # Tie the text LM head weights to the input embedding matrix
    tie_word_embeddings: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


_mha_block = lambda d_model=512, use_rotary_embedding=True, causal=True: MegaTransformerBlockConfig(
    d_model=d_model,
    n_heads=d_model // 64,
    d_queries=64,
    d_values=64,
    n_query_groups=d_model // 64,
    d_inner=d_model * 4,
    use_rotary_embedding=use_rotary_embedding,
    causal=causal,
)

WORLD_MODEL_CONFIGS = {
    # Original config: equal-weight preludes/codas/recurrent.
    "default": MegaTransformerWorldModelConfig(),

    "huginn_tiny_all_rotary": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(),
        ),
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(),
        ),
        # Heavy recurrent block: wide FFN, more heads, GQA
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=1024,
                n_heads=16,
                d_queries=64,
                d_values=64,
                n_query_groups=16,
                d_inner=4096,
                use_rotary_embedding=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            n_layers=2,
            coda_config=_mha_block(),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(),
            unpatchify_mode="pixel_shuffle",
        ),
    ),

    "huginn_tiny_text_sinusoidal_and_rotary": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(
            n_layers=2,
            use_pos_emb_ovr=True,
            prelude_config=MegaTransformerBlockConfig(
                d_model=512,
                n_heads=4,
                d_queries=64,
                d_values=64,
                n_query_groups=4,
                d_inner=1024,
                use_rotary_embedding=True,
            ),
        ),
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=1024,
                n_heads=16,
                d_queries=64,
                d_values=64,
                n_query_groups=16,
                d_inner=4096,
                use_rotary_embedding=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            n_layers=2,
            coda_config=_mha_block(),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(),
            unpatchify_mode="pixel_shuffle",
        ),
    ),

    "huginn_small_text_sinusoidal_and_rotary": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(
            n_layers=2,
            use_pos_emb_ovr=True,
            d_model=768,
            prelude_config=MegaTransformerBlockConfig(
                d_model=768,
                n_heads=4,
                d_queries=64,
                d_values=64,
                n_query_groups=4,
                d_inner=1024,
                use_rotary_embedding=True,
            ),
        ),
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=768 * 2,
                n_heads=12,
                d_queries=128,
                d_values=128,
                n_query_groups=12,
                d_inner=768 * 8,
                use_rotary_embedding=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            unpatchify_mode="pixel_shuffle",
        ),
    ),

    "huginn_small_bidirectional_preludes": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(
            n_layers=2,
            use_pos_emb_ovr=True,
            d_model=768,
            prelude_config=MegaTransformerBlockConfig(
                d_model=768,
                n_heads=4,
                d_queries=64,
                d_values=64,
                n_query_groups=4,
                d_inner=1024,
                use_rotary_embedding=True,
                causal=False,
            ),
        ),
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=768 * 2,
                n_heads=12,
                d_queries=128,
                d_values=128,
                n_query_groups=12,
                d_inner=768 * 8,
                use_rotary_embedding=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            unpatchify_mode="pixel_shuffle",
        ),
    ),

    "huginn_small_bidirectional_preludes_cross_attn_image_gen": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(
            n_layers=2,
            use_pos_emb_ovr=True,
            d_model=768,
            prelude_config=MegaTransformerBlockConfig(
                d_model=768,
                n_heads=4,
                d_queries=64,
                d_values=64,
                n_query_groups=4,
                d_inner=1024,
                use_rotary_embedding=True,
                causal=False,
            ),
        ),
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=768 * 2,
                n_heads=12,
                d_queries=128,
                d_values=128,
                n_query_groups=12,
                d_inner=768 * 8,
                use_rotary_embedding=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            unpatchify_mode="pixel_shuffle",
        ),
        image_cross_decoder_config={
            "decoder_config": {
                "d_model": 768,
                "n_heads": 12,
                "d_queries": 64,
                "d_values": 64,
                "n_query_groups": 12,
                "d_inner": 3072,
                "causal": False,
                "pre_attn_norm": True,
                "inter_attn_norm": True,
                "pre_ffn_norm": True,
                "use_rotary_embedding": False,
            },
            "n_encoder_layers": 4,
            "n_layers": 6,
            "latent_channels": 12,
            "latent_spatial_size": 32,
            "patch_size": 2,
        },
    ),

    # Same as huginn_small_bidirectional_preludes_cross_attn_image_gen but with
    # causal preludes for text and voice (matching autoregressive generation).
    # Image prelude stays bidirectional (only used for transcription; masked to
    # zeros for synthesis via is_synthesis flag). Cross-attention image decoder
    # handles image generation from content tokens.
    "huginn_small_causal_cross_attn": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(
            n_layers=2,
            use_pos_emb_ovr=True,
            d_model=768,
            prelude_config=MegaTransformerBlockConfig(
                d_model=768,
                n_heads=4,
                d_queries=64,
                d_values=64,
                n_query_groups=4,
                d_inner=1024,
                use_rotary_embedding=True,
                causal=True,
            ),
        ),
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=True),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=True),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            depth_scaled_init=False,
            iteration_norm="post_projection",
            block_config=MegaTransformerBlockConfig(
                d_model=768 * 2,
                n_heads=12,
                d_queries=128,
                d_values=128,
                n_query_groups=12,
                d_inner=768 * 8,
                use_rotary_embedding=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            unpatchify_mode="pixel_shuffle",
        ),
        image_cross_decoder_config={
            "decoder_config": {
                "d_model": 768,
                "n_heads": 12,
                "d_queries": 64,
                "d_values": 64,
                "n_query_groups": 12,
                "d_inner": 3072,
                "causal": False,
                "pre_attn_norm": True,
                "inter_attn_norm": True,
                "pre_ffn_norm": True,
                "use_rotary_embedding": False,
            },
            "n_encoder_layers": 4,
            "n_layers": 6,
            "latent_channels": 12,
            "latent_spatial_size": 32,
            "patch_size": 2,
        },
    ),
}
