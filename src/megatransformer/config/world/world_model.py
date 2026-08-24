import copy
import dataclasses
import json

from dataclasses import dataclass
from typing import Optional, List, Union

from megatransformer.config.voice.feature_extractor import VoiceSIVEPreludeFeatureExtractorConfig
from megatransformer.config.audio.generator import AudioCodaConfig
from megatransformer.config.voice.generator import VoiceCodaAndSMGConfig
from megatransformer.config.common import MegaTransformerBlockConfig
from megatransformer.config.image.decoder import (
    DiffusionBridgeImageDecoderConfig,
    ImageDecoderConfig,
    SDXLAdapterConfig,
    ZImageAdapterConfig,
)
from megatransformer.config.image.feature_extractor import ImageVAEPreludeFeatureExtractorConfig
from megatransformer.config.text.feature_extractor import TextPreludeFeatureExtractorConfig
from megatransformer.config.text.generator import TextCodaClassifierConfig
from megatransformer.utils.constants import (
    AUDIO_PLACEHOLDER_TOKEN_ID,
    VOICE_PLACEHOLDER_TOKEN_ID,
    IMAGE_PLACEHOLDER_TOKEN_ID,
    EOS_TOKEN_ID,
    special_token_ids,
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

    # Classifier-free guidance: create the learned null_text_embed parameter. Gated so a
    # non-CFG model has ZERO extra params -- adding a param unconditionally breaks true resumes
    # of pre-CFG checkpoints (optimizer-state param-group size mismatch). The trainer sets this
    # True when --voice_cfg_text_dropout_prob > 0.
    voice_cfg_enabled: bool = False
    # NAR (masked-parallel) voice synthesis. Builds the learned MASK feature the trainer
    # substitutes for masked frames; the coda must ALSO be made bidirectional
    # (voice_coda_config.coda_config.causal = False) or the head cannot see revealed units
    # to its right, which is the whole point of masked-parallel decoding.
    voice_nar: bool = False
    # NAR variant: the TRUNK sees only the MASK feature at voice positions (never revealed
    # units), and revealed units are routed straight into the coda instead. Confines any
    # inpainting shortcut to the head and keeps the trunk conditioned on text alone -- which
    # is where the measured crutch lives (trunk-level text-attributed fraction fell 0.220 at
    # r=1.0 to 0.029 at r=0.25 when revealed units reached it through the prelude).
    # Identical to voice_nar at mask ratio 1.0, since there are no revealed units to route.
    voice_nar_trunk_text_only: bool = False

    # Voice generation-query mode. None = the AR crutch (shifted-TF prelude output feeds the
    # recurrent trunk). "learned" = the trunk's SYNTHESIS voice input is a learned positional
    # gen query (text-only conditioning, crutch REMOVED from the shared trunk), and the coda
    # gets the previous-unit signal directly (local coherence stays at the coda, AR + EOV drive
    # length). Mirrors the image input/output split; input/transcription voice is unaffected.
    # Gated so a non-gen-query model has ZERO extra params (clean resumes). From-scratch only.
    voice_gen_query_mode: Optional[str] = None
    # Whether gen-query mode routes the previous frame's centroid to the coda (voice_coda_prev).
    # True = original behavior (a 1st-order neighbor crutch at the coda, survives even an MLP
    # coda). False = the coda sees ONLY the text-driven trunk output -> no neighbor crutch
    # anywhere. Only meaningful when voice_gen_query_mode is set.
    voice_gen_query_coda_prev: bool = True

    # SINGLE GATE for the pretrained-LLM text encoder. None (default) => the from-scratch
    # {wte + prelude transformer} path, byte-identical to current behavior. A dict opts in:
    #   {"model": "HuggingFaceTB/SmolLM2-135M", "freeze": True,
    #    "translator_hidden_mult": 2.0, "n_special_tokens": 0}
    # When set, the text prelude wraps the pretrained LLM body (+ input projection + MLP
    # translator) and the text coda becomes (MLP translator + the SHARED pretrained LM head).
    # The LLM owns its native vocab/tokenizer (frozen); "n_special_tokens" adds a small TRAINABLE
    # extension (tied special_embed/special_head) for control tokens (BOV/EOV/placeholders) at
    # ids >= native_vocab -- frozen-LLM-safe, no resize. Gated: non-users load unchanged.
    text_encoder: Optional[dict] = None
    # Base id for the 9 control tokens (BOA..IMAGE_PLACEHOLDER at base+0..8). = the real vocab
    # size: 32000 (Mistral, default) or the pretrained LLM's native vocab (set automatically in
    # pretrained mode). MUST match the base the data was tokenized with.
    # Nominal voice frames per text token for M-RoPE's LOCAL clock (25Hz speech vs SmolLM2
    # tokens is ~6). Only the ratio matters: it turns the text<->voice relationship from a
    # scaling (which RoPE cannot express) into a near-zero relative distance (which it can).
    mrope_voice_rate: float = 6.0
    # Which stream absorbs the rate: "voice" (voice t/rate — fractional sub-unit spacing for
    # the generated stream) or "text" (text j*rate — both streams stay integer-spaced).
    mrope_scale_side: str = "voice"
    special_token_base: int = 32_000
    # End-of-sequence id used to terminate text generation. Native to the vocab: 2 for the
    # Mistral tokenizer (default), or the pretrained LLM's native eos (set automatically in
    # pretrained mode). NOT one of the 9 control tokens.
    eos_token_id: int = EOS_TOKEN_ID

    # Feature extractor configs
    text_prelude_config: TextPreludeFeatureExtractorConfig = dataclasses.field(
        default_factory=TextPreludeFeatureExtractorConfig
    )
    audio_prelude_config: VoiceSIVEPreludeFeatureExtractorConfig = dataclasses.field(
        default_factory=VoiceSIVEPreludeFeatureExtractorConfig
    )
    voice_prelude_config: VoiceSIVEPreludeFeatureExtractorConfig = dataclasses.field(
        default_factory=VoiceSIVEPreludeFeatureExtractorConfig
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
    audio_coda_config: AudioCodaConfig = dataclasses.field(
        default_factory=AudioCodaConfig
    )
    voice_coda_config: VoiceCodaAndSMGConfig = dataclasses.field(
        default_factory=VoiceCodaAndSMGConfig
    )
    # Image decoder (optional). Two acceptable types:
    #   - `ImageDecoderConfig` for direct latent prediction (mode="direct" or
    #     "cross_attention"). Trained with whitened L1+MSE + variance losses.
    #   - `DiffusionBridgeImageDecoderConfig` for flow-matching DiT generation.
    #     Trained with flow-matching MSE; doesn't have the predict-the-mean
    #     attractor of the direct path.
    # None disables image generation entirely. The world model dispatches on
    # the actual config type to instantiate the right decoder class.
    image_coda_config: Optional[Union[ImageDecoderConfig, DiffusionBridgeImageDecoderConfig, SDXLAdapterConfig, ZImageAdapterConfig]] = None

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

    # Standard deviation of the initial image_gen_queries parameter (when
    # gen_query_mode="learned"). Historically 3.0, which made gen queries 3×
    # larger than the image prelude output (LayerNormed to std~1) and
    # comparable-to-slightly-larger than the text prelude output (~3.5). That
    # dominance may have prevented text self-attention updates inside the
    # recurrent block from overtaking the gen-query signal, keeping
    # prompt-conditional information low at the bridge input.
    #
    # 1.0 matches the image prelude output std so synthesis and transcription
    # positions enter the recurrent block at parity, and text-pulled updates
    # (each ~O(1/sqrt(n_blocks)) via depth-scaled residual init) compete on
    # equal footing from the first layer.
    image_gen_query_init_std: float = 3.0
    # LEVER D: subtract a LEARNED PER-POSITION offset from the trunk's gen-query output before
    # the image coda. The conditioning arrives as ~10% of a large per-position constant (the
    # learned gen queries, norm ~83, which survive the trunk largely intact). The existing
    # `image_coda_input_norm` CANNOT remove that: LayerNorm normalises each token across its
    # FEATURE axis and is shared across positions, so it only rescales a per-position vector --
    # measured at ckpt-85000 it takes cond/const 0.0991 -> 0.1136, about +15% relative.
    # Hypothesis: this is what `cross_dec` does implicitly (cross-attention with its own learned
    # queries can put anything it likes in the constant component), so doing it explicitly may
    # buy the same protection for ~49k params instead of 18.9M.
    # ⚠️ LEARNED, zero-init -- NOT the measured across-prompt mean. The head plausibly uses the
    # positional constant for slot IDENTITY (pos_spread/const is 0.97), and removing it wholesale
    # risks the positional collapse that `image_gen_query_init_std=3.0` was introduced to fix.
    # A learned offset can settle anywhere between identity and full cancellation.
    # Zero-init => bit-identical to off at step 0.
    image_gen_out_offset: bool = False

    def __post_init__(self):
        # Single source of truth for the control-token base: derive the interleaver's
        # placeholder ids from special_token_base so the collator, model, and interleaver
        # never disagree. With the default base (32000) these resolve to the same values
        # as the module-level constants -> byte-identical to prior behavior.
        sp = special_token_ids(self.special_token_base)
        tic = self.token_interleaver_config
        # None means "modality disabled" -- preserve it; only remap live ids.
        if tic.audio_placeholder_token_id is not None:
            tic.audio_placeholder_token_id = sp.AUDIO_PLACEHOLDER
        if tic.voice_placeholder_token_id is not None:
            tic.voice_placeholder_token_id = sp.VOICE_PLACEHOLDER
        if tic.image_placeholder_token_id is not None:
            tic.image_placeholder_token_id = sp.IMAGE_PLACEHOLDER

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
        audio_prelude_config=VoiceSIVEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        voice_prelude_config=VoiceSIVEPreludeFeatureExtractorConfig(
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
        audio_coda_config=AudioCodaConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            output_mode="conv_refine",
        ),
        voice_coda_config=VoiceCodaAndSMGConfig(
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
        audio_prelude_config=VoiceSIVEPreludeFeatureExtractorConfig(
            n_layers=2,
            prelude_config=_mha_block(d_model=768, causal=False),
        ),
        voice_prelude_config=VoiceSIVEPreludeFeatureExtractorConfig(
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
        audio_coda_config=AudioCodaConfig(
            n_layers=2,
            coda_config=_mha_block(d_model=768),
            output_mode="conv_refine",
        ),
        voice_coda_config=VoiceCodaAndSMGConfig(
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

# small_sum with the shared recurrent trunk at Huginn's depth (4 blocks) instead of 6.
# small_sum's 6 was a param-budget pick to round the FULL multimodal stack to ~300M; it was
# never depth-tuned, and Huginn used 4. Depth 4 drops the trunk 127.5M→85.0M (−42.5M): the
# full stack lands ~268M and a text→voice-only build ~184M (vs 226M), and it cuts the AR
# effective depth per token 32×6=192 → 32×4=128 (~33% faster generation). Derived from
# small_sum so everything BUT the recurrent depth stays identical and in sync. From-scratch
# only — existing checkpoints have 6 distinct blocks baked into their weights.
WORLD_MODEL_CONFIGS["small_sum_recd4"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum"])
WORLD_MODEL_CONFIGS["small_sum_recd4"].recurrent_block_config.n_recurrent_blocks = 4


# ── Frozen-SDXL image path: swap the DiT image coda for the SDXL conditioning
# adapter (predicts CLIP conditioning instead of a latent). The adapter is
# prelude-agnostic — it consumes the trunk's image gen-query outputs — so this
# composes with EITHER text prelude:
#   - from-scratch prelude:  text_encoder=None (inherited from small_sum below)
#   - SmolLM2 prelude:        set .text_encoder to the SAME dict your SmolLM2 runs
#     use (e.g. {"model": "HuggingFaceTB/SmolLM2-135M", "freeze": True, ...}) plus
#     the matching special_token_base/eos — must match how the data was tokenized.
# Only the image_coda_config changes here; everything else stays in sync with small_sum.
WORLD_MODEL_CONFIGS["small_sum_sdxl"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum"])
WORLD_MODEL_CONFIGS["small_sum_sdxl"].image_coda_config = SDXLAdapterConfig(
    d_model=768,          # match the recurrent/trunk d_model (small_sum uses 768)
    adapter_dim=768,
    n_heads=12,
    n_layers=2,
    n_cross_layers=2,
    contrastive_weight=1.0,
)

# Frozen Z-Image-Turbo path: predict Qwen3-4B conditioning instead of CLIP. Only the
# image_coda_config changes; everything else stays in sync with small_sum. BASELINE =
# pure MSE (contrastive_weight=0) to measure the naive target before any loss tricks.
WORLD_MODEL_CONFIGS["small_sum_zimage"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum"])
WORLD_MODEL_CONFIGS["small_sum_zimage"].image_coda_config = ZImageAdapterConfig(
    d_model=768,          # match the recurrent/trunk d_model (small_sum uses 768)
    adapter_dim=768,
    n_heads=12,
    n_layers=2,
    n_cross_layers=2,
    seq_len=64,
    contrastive_weight=0.0,
)

# Tier-0 whitened-MSE variant: identical to small_sum_zimage but regresses in per-dim
# z-scored Qwen3 space (needs --image_whiten_stats_path at train time; eval/chat use this
# same config so the adapter de-whitens from the checkpoint buffers).
WORLD_MODEL_CONFIGS["small_sum_zimage_whiten"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage"])
WORLD_MODEL_CONFIGS["small_sum_zimage_whiten"].image_coda_config.whiten_target = True

# Tier-1: whitened MSE + InfoNCE discriminability (ramped). Warm-start this from a whitened
# checkpoint (--resume_from_checkpoint <ckpt> --fresh_schedule); the projection head loads
# fresh (strict=False). contrastive_weight here is the RAMP TARGET (--image_contrastive_ramp_steps).
WORLD_MODEL_CONFIGS["small_sum_zimage_whiten_t1"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_whiten"])
WORLD_MODEL_CONFIGS["small_sum_zimage_whiten_t1"].image_coda_config.contrastive_weight = 0.2

# Tier-1q: same, plus a 4096-entry memory queue of past Qwen3 targets. Plain in-batch
# InfoNCE at batch_size 8 is an 8-way task that the adapter solves to ~0.009 (chance =
# ln 8 = 2.08) within a few hundred steps, so the term contributes essentially no
# gradient; the queue makes it 4104-way (chance = ln 4104 = 8.3) and keeps it hard for
# the whole run. Queue is non-persistent, so checkpoints stay the same size and this
# config still loads t1 checkpoints (and vice versa).
WORLD_MODEL_CONFIGS["small_sum_zimage_whiten_t1q"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_whiten_t1"])
WORLD_MODEL_CONFIGS["small_sum_zimage_whiten_t1q"].image_coda_config.contrastive_queue_size = 4096

# Tier-3: whitened flow-matching SAMPLER over the Qwen3 conditioning (no InfoNCE -- T1 was
# measured to improve retrieval only, +0.003 R^2, render flat). Fixes the under-dispersion that
# a point loss is structurally forced into, per-caption, instead of patching it with a global
# output_gain. Warm-start from a whitened regression checkpoint (--resume_from_checkpoint
# <whiten or t1q ckpt> --fresh_schedule): the Q-Former/trunk load, the flow head starts fresh.
WORLD_MODEL_CONFIGS["small_sum_zimage_t3"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_whiten"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t3"].image_coda_config.flow_head = True

# T4: AUTOREGRESSIVE conditioning with a per-token flow head (native length, no resample).
# Same whitening + CFG dropout as T3; the parallel flow head is replaced by the AR one, so
# the image arm factorises its output the way the text and voice arms do -- one mechanism
# from the shared trunk. NOTE: the K=64 resample this removes was measured to cost ~0.0004
# CLIPScore, so this is an architectural-uniformity change, not a fix for a measured gap.
WORLD_MODEL_CONFIGS["small_sum_zimage_t4_ar"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_whiten"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t4_ar"].image_coda_config.ar_flow_head = True

# T5: T3's PARALLEL head at NATIVE length. T4 bundled two changes -- autoregressive
# factorisation AND native length -- and lost decisively (w=3 plateau ~0.248 vs T3's 0.344),
# which indicts the factorisation, not the length handling. This keeps the winning parallel
# sampler and drops only the K=64 resample, so ~2/3 of the supervised slots stop being linear
# blends of neighbouring Qwen3 states. Inference stays one-shot (no sequential decode).
# Warm-start from a T3 checkpoint: trunk/Q-Former/flow head all load, since the head's shape is
# unchanged -- only how many slots it is asked for.
WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t3"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native"].image_coda_config.flow_native_length = True
WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native"].image_coda_config.flow_aux_mse_weight = 0.0

# T5 + slot positions. Separate preset so "native length" and "the slots get an identity" are
# ablatable independently -- without pos_embed the head is permutation-equivariant and emits an
# unordered set (which is how T3 works today).
WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native_pos"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native_pos"].image_coda_config.flow_pos_embed = True

# T3 + the x_t skip that closes the rank-limited-output noise leak. This is a FIX to T3's head,
# not a new tier: same parallel sampler, same K=64 targets, same everything -- plus the full-rank
# -x_t/(1-t) term the rectified-flow interpolant actually calls for. Zero-init, so it warm-starts
# from t3_2 bit-identically and learns the skip from there.
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t3"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip"].image_coda_config.flow_x_skip = True

# T5 (native length) + the same skip -- the two independent fixes composed.
WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native_xskip"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native_xskip"].image_coda_config.flow_x_skip = True

# T5 native + xskip + SLOT POSITIONS. Measured 2026-08-20: the native+xskip arm renders coherent
# COMPOSITION but overlays structured, semantically empty blob texture -- the off-manifold
# signature, not the bland/under-dispersed one (its whitened std is 0.856, under-dispersed, which
# on its own predicts BLANDNESS; the blobs are something else).
# Suspected cause: without `pos` the parallel head is exactly permutation-equivariant -- it emits
# an unordered SET. At K=64 that survived because the target was ~20 real Qwen3 states smoothly
# interpolated into 64 redundant slots, so order washed out. At NATIVE length every token is
# distinct and load-bearing, and Qwen3 states are CAUSAL (token j encodes everything up to j), so
# a set with no positional identity cannot reliably form a valid ordered prefix sequence.
# => native length and slot identity are NOT independent knobs; this preset couples them.
WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native_pos_xskip"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native_xskip"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t5_native_pos_xskip"].image_coda_config.flow_pos_embed = True

# T3 + x1-PREDICTION: the fully grounded form of the noise-leak fix. Same parallel head, same K=64
# resampled targets (which BEAT native length by 0.045), same CFG -- only the velocity
# parameterisation changes: predict x1, derive v = (x1_hat - x_t)/(1-t). The learned x_skip left
# 44.8% of the initial noise alive because gradient descent will not find a near-singular
# coefficient; this applies it analytically. Warm-starts from t3_2 or from a t3_xskip checkpoint
# (skip_ada is simply unused).
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_x1pred"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t3"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_x1pred"].image_coda_config.flow_x1_pred = True
# Min-SNR weighting is NOT optional here: without it the ~1.4% of logit-normal samples above
# t=0.9 carry 76-1887x the gradient and, under max_grad_norm clipping, dominate the update.
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_x1pred"].image_coda_config.flow_loss_weighting = "min_snr"

# Same, but with the FLATTER weighting: multiply by (1-t)^2, i.e. a uniform MSE in x1 space.
# Offline gradient profile across t: velocity ~1x, x1+min_snr ~7x, x1+x1-space ~1.1x. In the
# min_snr run, clipping (max_grad_norm 1.0) peaked at 60% of steps around 500-1000 before decaying
# to 0% by step 1500 -- so the mid-schedule third trained under truncated updates. This removes
# the late-t emphasis entirely rather than capping it.
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_x1pred_x1w"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t3_x1pred"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_x1pred_x1w"].image_coda_config.flow_loss_weighting = "x1"

# Q-FORMER ABLATION: condition the flow head on the trunk states directly. Tests whether
# cross_dec (18.9M params) is earning its keep on a conditioning path already suspected of being
# over-constricted, or is a redundant second cross-attention in front of the head's own.
# ⚠️ TEST FROM SCRATCH: a warm start inherits a trunk co-adapted to HAVING a Q-Former, the same
# debt that made the native-length warm-start uninterpretable.
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_trunkctx"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_trunkctx"].image_coda_config.flow_ctx = "trunk"

# cross_dec DROPPED OUTRIGHT -- the real "is the Q-Former necessary" arm.
# `_trunkctx` above does NOT answer that: zimage_adapter.py runs cross_dec unconditionally and
# seq_pred always reads its output, so there the Q-Former is still built, still executed and
# still trained by the 0.1 aux MSE; only the FLOW HEAD's context changes. Here the module and
# its out_queries are never constructed, so they are gone from every forward and every gradient
# path, and the aux point head reads the self_enc'd trunk states instead (so alpha / R^2 /
# retrieval diagnostics still work). Valid because n_image_gen_positions == seq_len == 64, which
# makes cross_dec's K -> seq_len remap identity-shaped anyway; the adapter raises if they differ.
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_nocrossdec"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_nocrossdec"].image_coda_config.use_cross_dec = False

# LEVER D: learned per-position centering of the trunk's gen-query output (see
# `image_gen_out_offset`). Tests whether an explicit constant-remover buys what `cross_dec`
# appears to buy implicitly, for ~49k params instead of 18.9M.
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_offset"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_offset"].image_gen_out_offset = True

# The combination that actually tests the hypothesis: drop cross_dec AND add the explicit
# centering. If the constant-remover reading is right this should recover most of what
# `_nocrossdec` alone loses, at 1/380th the parameter cost.
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_nocrossdec_offset"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_nocrossdec"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_nocrossdec_offset"].image_gen_out_offset = True

# xskip + explicit DISPERSION MATCHING. The best arm so far (xskip) still emits under-dispersed
# content and leans on both leftover noise and CFG to make up for it; this makes the spread an
# objective instead. Weight 0.5 with a small anti-collapse barrier: the term is dimensionless and
# ~1 at full collapse, so it is comparable in scale to the flow loss itself.
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_var"] = copy.deepcopy(WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip"])
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_var"].image_coda_config.flow_var_loss_weight = 0.5
WORLD_MODEL_CONFIGS["small_sum_zimage_t3_xskip_var"].image_coda_config.flow_var_barrier_weight = 0.1
