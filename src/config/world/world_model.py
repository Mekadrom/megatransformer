import dataclasses
import json

from dataclasses import dataclass
from typing import Optional, List, Union

from config.audio.feature_extractor import AudioVAEPreludeFeatureExtractorConfig
from config.audio.generator import AudioCodaAndVAEConfig
from config.common import MegaTransformerBlockConfig
from config.image.decoder import (
    DiffusionBridgeImageDecoderConfig,
    ImageDecoderConfig,
)
from config.image.feature_extractor import ImageVAEPreludeFeatureExtractorConfig
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
    block_init_gain: float = 1.0  # Xavier gain for recurrent block weights (1.0 = standard Xavier; depth_scaled_init handles residual stability)
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
    # Image decoder (optional). Two acceptable types:
    #   - `ImageDecoderConfig` for direct latent prediction (mode="direct" or
    #     "cross_attention"). Trained with whitened L1+MSE + variance losses.
    #   - `DiffusionBridgeImageDecoderConfig` for flow-matching DiT generation.
    #     Trained with flow-matching MSE; doesn't have the predict-the-mean
    #     attractor of the direct path.
    # None disables image generation entirely. The world model dispatches on
    # the actual config type to instantiate the right decoder class.
    image_coda_config: Optional[Union[ImageDecoderConfig, DiffusionBridgeImageDecoderConfig]] = None

    # Scale embeddings by sqrt(d_model) before recurrent block (Huginn-style)
    scale_embeddings: bool = False

    # Tie the text LM head weights to the input embedding matrix
    tie_word_embeddings: bool = False

    # Generation query mode for IMAGE synthesis only. Voice/audio use
    # autoregressive generation (shifted teacher forcing at train time,
    # coda-prediction re-encoding at inference) — no gen queries needed.
    #   "learned" — learned nn.Parameter + frozen sinusoidal PE (default)
    #   "positional_only" — frozen sinusoidal PE only, no learned component
    gen_query_mode: str = "learned"

    # Number of image generation query positions for synthesis (text → image).
    # Controls how many image tokens appear in the interleaved sequence during
    # synthesis — i.e., how many "slots" the recurrent block gets to fill with
    # image-relevant content before the bridge compresses them to DiT conditioning.
    #
    # Must be a perfect square (the 2D sinusoidal PE encodes row/col on a grid).
    # Common values: 16 (4×4), 36 (6×6), 64 (8×8), 144 (12×12), 256 (16×16).
    #
    # None (default) = use the image prelude's patch count for backward compat.
    # Set explicitly to decouple synthesis from transcription tokenization — the
    # bridge is sequence-length agnostic so any value works.
    n_image_gen_positions: Optional[int] = None

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


def _image_decoder(
    d_model: int = 768,
    *,
    mode: str = "direct",
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 6,
    d_inner: int = 3072,
    latent_channels: int = 12,
    latent_spatial_size: int = 32,
    patch_size: int = 2,
    use_output_denorm: bool = True,
) -> ImageDecoderConfig:
    """Build an ImageDecoderConfig with sensible defaults for the small configs."""
    return ImageDecoderConfig(
        mode=mode,
        block_config=MegaTransformerBlockConfig(
            d_model=d_model,
            n_heads=d_model // 64,
            d_queries=64,
            d_values=64,
            n_query_groups=d_model // 64,
            d_inner=d_inner,
            causal=False,
            pre_attn_norm=True,
            inter_attn_norm=True,
            pre_ffn_norm=True,
            use_rotary_embedding=False,
        ),
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        latent_channels=latent_channels,
        latent_spatial_size=latent_spatial_size,
        patch_size=patch_size,
        use_output_denorm=use_output_denorm,
    )

WORLD_MODEL_CONFIGS = {
    # Original config: equal-weight preludes/codas/recurrent.
    "default": MegaTransformerWorldModelConfig(),

    "small_sum": MegaTransformerWorldModelConfig(
        gen_query_mode='learned',
        text_prelude_config=TextPreludeFeatureExtractorConfig(
            n_layers=2,
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
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=True),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
            # LiteVAE latents arrive at the prelude with std much larger than 1
            # (range roughly [-7.5, 7.5]). Output norm caps the prelude output
            # at std~1 before it reaches the recurrent block, so image positions
            # don't dominate text positions in the interleaved sequence.
            use_output_norm=True,
            output_norm_type="layernorm",
        ),
        recurrent_block_config=MegaTransformerRecurrentConfig(
            n_recurrent_blocks=6,
            injection_type='add',
            depth_scaled_init=True,
            iteration_norm="post_projection",
            block_config=MegaTransformerBlockConfig(
                d_model=768,
                n_heads=12,
                d_queries=64,
                d_values=64,
                n_query_groups=12,
                d_inner=768 * 16,
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
            output_mode="framewise_refine",
        ),
        image_coda_config=DiffusionBridgeImageDecoderConfig(
            # Match the recurrent block's d_model so the bridge doesn't need a
            # cross-d_model projection at its input.
            d_model=768,
            # Bridge: 4 layers with 64 learnable queries cross-attending to the
            # recurrent block's image-position outputs.
            n_bridge_layers=2,
            n_bridge_queries=64,
            # DiT: 4 layers — comparable depth to the small_sum recurrent block
            # so the parameter budget is similar to the direct decoder it replaces.
            n_dit_layers=4,
            # Match the LiteVAE latent shape from the existing image dataset.
            latent_channels=12,
            latent_spatial_size=32,
            patch_size=2,
            # SD3/Flux-style timestep distribution: biases mid-range timesteps
            # which are typically the most informative to train on.
            timestep_sampling="logit_normal",
            # Inference: Euler integration steps. 16 is conservative for
            # from-scratch training; can be lowered after the model has trained.
            num_inference_steps=16,
        ),
    ),

    "small_concat": MegaTransformerWorldModelConfig(
        gen_query_mode='positional_only',
        text_prelude_config=TextPreludeFeatureExtractorConfig(
            n_layers=2,
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
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        voice_prelude_config=AudioVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=True),
        ),
        image_prelude_config=ImageVAEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
            # LiteVAE latents arrive at the prelude with std much larger than 1
            # (range roughly [-7.5, 7.5]). Output norm caps the prelude output
            # at std~1 before it reaches the recurrent block, so image positions
            # don't dominate text positions in the interleaved sequence.
            use_output_norm=True,
            output_norm_type="layernorm",
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
                d_inner=768 * 16,
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
            output_mode="framewise_refine",
        ),
        image_coda_config=DiffusionBridgeImageDecoderConfig(
            # Match the recurrent block's d_model so the bridge doesn't need a
            # cross-d_model projection at its input.
            d_model=768,
            # Bridge: 4 layers with 64 learnable queries cross-attending to the
            # recurrent block's image-position outputs.
            n_bridge_layers=2,
            n_bridge_queries=64,
            # DiT: 4 layers — comparable depth to the small_sum recurrent block
            # so the parameter budget is similar to the direct decoder it replaces.
            n_dit_layers=4,
            # Match the LiteVAE latent shape from the existing image dataset.
            latent_channels=12,
            latent_spatial_size=32,
            patch_size=2,
            # SD3/Flux-style timestep distribution: biases mid-range timesteps
            # which are typically the most informative to train on.
            timestep_sampling="logit_normal",
            # Inference: Euler integration steps. 16 is conservative for
            # from-scratch training; can be lowered after the model has trained.
            num_inference_steps=16,
        ),
    ),

}
