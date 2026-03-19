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

    # Scale embeddings by sqrt(d_model) before recurrent block (Huginn-style)
    scale_embeddings: bool = False

    # Tie the text LM head weights to the input embedding matrix
    tie_word_embeddings: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Slim prelude/coda block: lightweight adapter — just enough to translate
# between modality-specific representations and the shared d_model space.
_slim_block = lambda: MegaTransformerBlockConfig(
    d_model=512,
    n_heads=4,
    d_queries=64,
    d_values=64,
    n_query_groups=2,
    d_inner=1024,
)

_mha_block = lambda: MegaTransformerBlockConfig(
    d_model=512,
    n_heads=4,
    d_queries=64,
    d_values=64,
    n_query_groups=4,
    d_inner=1024,
    use_rotary_embedding=True,
)

WORLD_MODEL_CONFIGS = {
    # Original config: equal-weight preludes/codas/recurrent.
    "default": MegaTransformerWorldModelConfig(),

    "test": MegaTransformerWorldModelConfig(
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
        # Slim codas
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

    # Researcher config: recurrent block dominates (~40-45% of params).
    # Preludes and codas use slim single-block transformers.
    # The recurrent block gets a wide FFN (12288) and 16 GQA heads,
    # since it iterates many times per token — every parameter is reused.
    "researcher": MegaTransformerWorldModelConfig(
        # Slim preludes (image uses layernorm, not instancenorm — original working config)
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
            input_norm_type="layernorm",
        ),
        # Heavy recurrent block: wide FFN, more heads, GQA
        recurrent_block_config=MegaTransformerRecurrentConfig(
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=1024,      # 2 * base d_model (concat input + thought)
                n_heads=16,
                d_queries=64,
                d_values=64,
                n_query_groups=4,  # GQA: 4 KV groups for 16 heads
                d_inner=12288,     # wide FFN — bulk of the parameters
            ),
        ),
        # Slim codas
        text_coda_config=TextCodaClassifierConfig(
            coda_config=_slim_block(),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=_slim_block(),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=_slim_block(),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=_slim_block(),
            unpatchify_mode="pixel_shuffle",
        ),
    ),

    # ~72M — researcher with additive injection (Huginn-style). Same preludes/codas
    # as researcher, but recurrent block operates at d=512 (additive, no concat/projection).
    # Massive FFN ratio (52x) because attention at d=512 is tiny.
    "researcher_add": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=512),
        scale_embeddings=False,
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            block_config=MegaTransformerBlockConfig(
                d_model=512, n_heads=8, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=26752,
            ),
            injection_type="add",
        ),
        text_coda_config=TextCodaClassifierConfig(
            coda_config=_slim_block(),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=_slim_block(),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=_slim_block(),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=_slim_block(),
            unpatchify_mode="pixel_shuffle",
        ),
    ),

    # ~75M — 200m_safe but with 1 recurrent block. Ablation test to isolate
    # whether 4-block cycling is the memorization bottleneck.
    # ~200M — 2-block variant of 200m_safe. Tests whether 2 sequential blocks per
    # iteration work where 4 failed (gradient depth issue).
    "200m_safe_2blk": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=640),
        scale_embeddings=False,
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=2,
            block_config=MegaTransformerBlockConfig(
                d_model=1280, n_heads=20, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=29568,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),

    # ~75M — additive injection at d=640 with instancenorm (default).
    # Tests whether additive injection works at this base dimension.
    "researcher_add_640": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=640),
        scale_embeddings=False,
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=1,
            injection_type="add",
            block_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=26624,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),

    "200m_safe_1blk": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=640),
        scale_embeddings=False,
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=1,
            block_config=MegaTransformerBlockConfig(
                d_model=1280, n_heads=20, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=14016,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),

    # ~200M params — scaled researcher. Identical structure to researcher (LayerNorm,
    # pre-norm, GQA=4, 1-layer preludes/codas) but with 4 cycled recurrent blocks
    # and wider FFN. Use this as a known-good baseline for ablation.
    "200m_safe": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=640),
        scale_embeddings=False,
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            block_config=MegaTransformerBlockConfig(
                d_model=1280, n_heads=20, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=14016,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),

    # 200m_safe with sandwich norm (pre+post attn, pre+post FFN) in recurrent blocks only.
    # Preludes/codas stay at standard pre-norm. Tests Huginn's approach to
    # preventing activation growth in the 4-block recurrent chain.
    "200m_safe_sandwich": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=640),
        scale_embeddings=False,
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=1280, n_heads=20, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=14016,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=640, n_heads=10, d_queries=64, d_values=64,
                n_query_groups=2, d_inner=1280,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),

    # ~95M — researcher with wider recurrent FFN (32x). Tests whether
    # researcher architecture scales by just widening FFN at d=512.
    "researcher_wide": MegaTransformerWorldModelConfig(
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=_slim_block(),
            input_norm_type="layernorm",
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=32768,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            coda_config=_slim_block(),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=_slim_block(),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=_slim_block(),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=_slim_block(),
            unpatchify_mode="pixel_shuffle",
        ),
    ),

    # ~101M params (text+voice+image). Standard FFN ratios throughout.
    # base d=768, recurrent d=1536 (concat), 1 block.
    # Preludes/codas: 4x FFN. Recurrent: 8x FFN.
    "small": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=768),
        scale_embeddings=False,
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=3, d_inner=3072,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=3, d_inner=3072,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=3, d_inner=3072,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=1,
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=1536, n_heads=24, d_queries=64, d_values=64,
                n_query_groups=6, d_inner=12288,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=3, d_inner=3072,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=3, d_inner=3072,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=3, d_inner=3072,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=3, d_inner=3072,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),

    # ~169M params (text+voice+image). Standard FFN ratios throughout.
    # base d=1024, recurrent d=2048 (concat), 1 block.
    # Preludes/codas: 4x FFN. Recurrent: 8x FFN.
    "medium": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=1024),
        scale_embeddings=False,
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=4096,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=4096,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            prelude_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=4096,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=1,
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=2048, n_heads=32, d_queries=64, d_values=64,
                n_query_groups=8, d_inner=16384,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=4096,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=4096,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=4096,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            coda_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=4096,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),

    # ~1B params (with tie_word_embeddings, text+voice+image).
    # Matches 200m design: 2-layer preludes/codas (pre-norm only), 4 cycled recurrent
    # blocks (sandwich RMSNorm), no GQA in recurrent (full MHA), no depth-scaled init,
    # no embed scaling. All FFN ratios >= 4x: preludes/codas 4.0x, recurrent 5.9x.
    # base d_model=1536, recurrent d_model=3072 (concat).
    "1b": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=1536),
        scale_embeddings=False,
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=1536, n_heads=24, d_queries=64, d_values=64,
                n_query_groups=6, d_inner=6144,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=1536, n_heads=24, d_queries=64, d_values=64,
                n_query_groups=6, d_inner=6144,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=1536, n_heads=24, d_queries=64, d_values=64,
                n_query_groups=6, d_inner=6144,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            depth_scaled_init=False,
            block_config=MegaTransformerBlockConfig(
                d_model=3072, n_heads=48, d_queries=64, d_values=64,
                n_query_groups=48, d_inner=18176,
                norm_type="rmsnorm", norm_eps=1e-6,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1536, n_heads=24, d_queries=64, d_values=64,
                n_query_groups=6, d_inner=6144,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1536, n_heads=24, d_queries=64, d_values=64,
                n_query_groups=6, d_inner=6144,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1536, n_heads=24, d_queries=64, d_values=64,
                n_query_groups=6, d_inner=6144,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1536, n_heads=24, d_queries=64, d_values=64,
                n_query_groups=6, d_inner=6144,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),

    # ~200M params, additive injection variant for A/B testing against "200m" (concat).
    # Same preludes/codas as "200m", but recurrent blocks operate at d_model=1024
    # (additive injection, Huginn-style) instead of 2*d_model=2048 (concat).
    # The freed attention budget goes to FFN: 9.8× ratio vs 0.9× in "200m".
    "200m_add": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=1024),
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=1024,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=1024,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=1024,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            thought_init_std=0.6325,
            injection_type="add",
            block_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=16, d_inner=10112,
                norm_type="rmsnorm", norm_eps=1e-6,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=1024,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=1024,
                post_attn_norm=True, post_ffn_norm=True,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=1024,
                post_attn_norm=True, post_ffn_norm=True,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1024, n_heads=16, d_queries=64, d_values=64,
                n_query_groups=4, d_inner=1024,
                post_attn_norm=True, post_ffn_norm=True,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),

    # ~200M, full MHA variant — no GQA anywhere (preludes, codas, or recurrent).
    # Otherwise identical to "200m". FFN ratio drops from 5.2x to 4.5x due to
    # larger attention projections in preludes/codas.
    "200m_mha": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=768),
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=12, d_inner=768,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=12, d_inner=768,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=12, d_inner=768,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            thought_init_std=0.6325,
            block_config=MegaTransformerBlockConfig(
                d_model=1536, n_heads=24, d_queries=64, d_values=64,
                n_query_groups=24, d_inner=6960,
                norm_type="rmsnorm", norm_eps=1e-6,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=12, d_inner=768,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=12, d_inner=768,
                post_attn_norm=True, post_ffn_norm=True,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=12, d_inner=768,
                post_attn_norm=True, post_ffn_norm=True,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=768, n_heads=12, d_queries=64, d_values=64,
                n_query_groups=12, d_inner=768,
                post_attn_norm=True, post_ffn_norm=True,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),

    # ~1B, full MHA variant — no GQA anywhere. Otherwise identical to "1b".
    # FFN ratio drops from 5.1x to 4.5x.
    "1b_mha": MegaTransformerWorldModelConfig(
        text_prelude_config=TextPreludeFeatureExtractorConfig(d_model=1792),
        audio_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=1792, n_heads=28, d_queries=64, d_values=64,
                n_query_groups=28, d_inner=1792,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=1792, n_heads=28, d_queries=64, d_values=64,
                n_query_groups=28, d_inner=1792,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=MegaTransformerBlockConfig(
                d_model=1792, n_heads=28, d_queries=64, d_values=64,
                n_query_groups=28, d_inner=1792,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=4,
            thought_init_std=0.6325,
            block_config=MegaTransformerBlockConfig(
                d_model=3584, n_heads=56, d_queries=64, d_values=64,
                n_query_groups=56, d_inner=15971,
                norm_type="rmsnorm", norm_eps=1e-6,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        text_coda_config=TextCodaClassifierConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1792, n_heads=28, d_queries=64, d_values=64,
                n_query_groups=28, d_inner=1792,
                post_attn_norm=True, post_ffn_norm=True,
            ),
        ),
        audio_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1792, n_heads=28, d_queries=64, d_values=64,
                n_query_groups=28, d_inner=1792,
                post_attn_norm=True, post_ffn_norm=True,
            ),
            output_mode="conv_refine",
        ),
        voice_coda_config=AudioCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1792, n_heads=28, d_queries=64, d_values=64,
                n_query_groups=28, d_inner=1792,
                post_attn_norm=True, post_ffn_norm=True,
            ),
            output_mode="conv_refine",
        ),
        image_coda_config=ImageCodaAndVAEConfig(
            n_layers=2,
            coda_config=MegaTransformerBlockConfig(
                d_model=1792, n_heads=28, d_queries=64, d_values=64,
                n_query_groups=28, d_inner=1792,
                post_attn_norm=True, post_ffn_norm=True,
            ),
            unpatchify_mode="pixel_shuffle",
        ),
        tie_word_embeddings=True,
    ),
}
