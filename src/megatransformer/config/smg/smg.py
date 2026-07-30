import dataclasses
import json

from dataclasses import dataclass
from typing import Optional


@dataclass
class SMGDecoder2DConfig:
    """Configuration for 2D SMG (SIVE-Mel Generator) decoder."""

    # architecture parameters
    # Channel width of the SIVE encoder features fed into the decoder (the
    # decoder input dim). Defaults to the current SIVE encoder_dim (256).
    sive_encoder_dim: int = 256
    # InstanceNorm the RAW SIVE features (per-channel over time) before the initial
    # conv — strips per-channel utterance moments (speaker envelope) at the input,
    # forcing that detail to come from the embedding (AutoVC/IN-VC style).
    # Architectural, affine-free, off by default.
    input_instance_norm: bool = False

    activation: str = "silu"

    intermediate_channels: Optional[list[int]] = None
    kernel_sizes: Optional[list[int]] = None
    strides: Optional[list[int]] = None
    n_residual_blocks: int = 1
    speaker_embedding_dim: int = 192
    speaker_embedding_proj_dim: int = 0
    normalize_speaker_embedding: bool = True
    film_scale_bound: float = 1.0
    film_shift_bound: float = 1.0
    zero_init_film_bias: bool = True
    f0_embedding_dim: int = 64

    output_dim: int = 80  # n_mels

    conv1d_channels: int = 320
    conv1d_n_residual_blocks: int = 2
    conv1d_kernel_size: int = 5
    conv1d_upsample_factor: int = 1

    initial_freq_bins: int = 10  # when reshape from 1D to 2D feature map

    intermediate_channels_2d: Optional[list[int]] = None
    kernel_sizes_2d: Optional[list[tuple[int, int]]] = None
    scale_factors_2d: Optional[list[tuple[int, int]]] = None
    n_residual_blocks_2d: int = 2

    # training parameters
    dropout: float = 0.1
    logvar_clamp_max: float = 4.0


    def __post_init__(self):
         # defaults for 2D stages
        if self.intermediate_channels_2d is None:
            self.intermediate_channels_2d = [64, 128, 256]
        if self.kernel_sizes_2d is None:
            self.kernel_sizes_2d = [(3, 5), (3, 5), (3, 5)]
        if self.scale_factors_2d is None:
            # Default: 8x freq (10→80), 4x time (T'→4T')
            self.scale_factors_2d = [(2, 2), (2, 2), (2, 1)]

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


SMG_DECODER_2D_CONFIGS = {
    "default": SMGDecoder2DConfig(),
    "small": SMGDecoder2DConfig(
        intermediate_channels_2d=[64, 96, 144],
        kernel_sizes_2d=[(3, 5), (3, 5), (3, 5)],
        scale_factors_2d=[(2, 2), (2, 2), (2, 1)],  # 8x freq, 4x time
    ),
}


@dataclass
class SMGDecoder1DConfig:
    """Configuration for Conv1D-only SMG (SIVE-Mel Generator) decoder.

    Treats frequency as channels throughout — no 1D→2D reshape.
    Avoids ring artifacts caused by Conv2D spatial isotropy assumption.
    """

    # Channel width of the SIVE encoder features fed into the decoder (the
    # decoder input dim). Defaults to the current SIVE encoder_dim (256).
    sive_encoder_dim: int = 256
    # InstanceNorm the RAW SIVE features (per-channel over time) before the initial
    # conv — strips the per-channel utterance moments (a speaker slice: global
    # spectral envelope) at the SMG's input, forcing that detail to come from the
    # embedding instead (AutoVC/IN-VC style). Architectural (not baked into the
    # cache), affine-free, off by default; toggling it reuses the same features.
    input_instance_norm: bool = False
    activation: str = "silu"

    # Speaker conditioning (FiLM)
    speaker_embedding_dim: int = 192
    speaker_embedding_proj_dim: int = 0
    normalize_speaker_embedding: bool = True
    film_scale_bound: float = 1.0
    film_shift_bound: float = 1.0
    zero_init_film_bias: bool = True

    # F0 conditioning
    f0_embedding_dim: int = 64

    # 1D architecture — frequency is channels
    initial_channels: int = 512
    stage_channels: Optional[list[int]] = None
    stage_kernel_sizes: Optional[list[int]] = None
    time_upsample_factors: Optional[list[int]] = None
    # Optional integer time DOWNSAMPLE applied once after the upsample stages (a strided
    # Conv1d), so the net rate can be a non-integer like 7.5x WITHOUT any fractional
    # resample of the output mel: e.g. upsample 15x ([3,5]) then downsample 2x -> exactly
    # 12.5Hz*15/2 = 93.75Hz (Vocos), landing on mel_length +-1 frame (handled by the
    # replicate-pad in SMG.forward, never F.interpolate). 1 = no downsample (default).
    time_downsample_factor: int = 1
    n_residual_blocks_per_stage: int = 2
    pre_upsample_residual_blocks: int = 2
    pre_upsample_kernel_size: int = 5

    output_dim: int = 80  # n_mels
    dropout: float = 0.1

    def __post_init__(self):
        if self.stage_channels is None:
            self.stage_channels = [512, 256, 256]
        if self.stage_kernel_sizes is None:
            self.stage_kernel_sizes = [5, 5, 5]
        if self.time_upsample_factors is None:
            self.time_upsample_factors = [2, 2, 1]  # 4x total time upsampling

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


SMG_DECODER_1D_CONFIGS = {
    "default": SMGDecoder1DConfig(),
    "small": SMGDecoder1DConfig(
        initial_channels=512,
        stage_channels=[512, 256, 256],
        stage_kernel_sizes=[5, 5, 5],
        time_upsample_factors=[2, 2, 1],  # 4x total
    ),
    "medium_1x": SMGDecoder1DConfig(
        initial_channels=640,
        stage_channels=[640, 384, 256],
        stage_kernel_sizes=[7, 5, 5],
        time_upsample_factors=[1, 1, 1],  # 1x total (ContentVec@50 = mel@50, no time upsampling)
        n_residual_blocks_per_stage=3,
        pre_upsample_residual_blocks=3,
        pre_upsample_kernel_size=7,
    ),
    "medium_3x": SMGDecoder1DConfig(
        initial_channels=640,
        stage_channels=[640, 384, 256],
        stage_kernel_sizes=[7, 5, 5],
        time_upsample_factors=[3, 1, 1],  # 3x total
        n_residual_blocks_per_stage=3,
        pre_upsample_residual_blocks=3,
        pre_upsample_kernel_size=7,
    ),
    "medium_4x": SMGDecoder1DConfig(
        initial_channels=640,
        stage_channels=[640, 384, 256],
        stage_kernel_sizes=[7, 5, 5],
        time_upsample_factors=[2, 2, 1],  # 4x total
        n_residual_blocks_per_stage=3,
        pre_upsample_residual_blocks=3,
        pre_upsample_kernel_size=7,
    ),
    # Mimi 12.5Hz -> 8x -> 100Hz, then SMG.forward resamples to the Vocos target rate
    # (93.75Hz @ hop256/24kHz; the 12.5->93.75 ratio is 7.5x, non-integer, so we
    # over-upsample to a clean 8x and interpolate the ~6.7% down to mel_length).
    # output_dim=100 = Vocos's mel bands.
    "medium_8x_100mel": SMGDecoder1DConfig(
        initial_channels=640,
        stage_channels=[640, 384, 256],
        stage_kernel_sizes=[7, 5, 5],
        time_upsample_factors=[2, 2, 2],  # 8x total
        n_residual_blocks_per_stage=3,
        pre_upsample_residual_blocks=3,
        pre_upsample_kernel_size=7,
        output_dim=100,
    ),
    # Mimi 12.5Hz -> EXACT 93.75Hz (Vocos) with NO fractional resample: upsample 15x ([3,5])
    # then a strided-conv downsample 2x = net 7.5x. Output lands on mel_length +-1 frame
    # (replicate-pad in SMG.forward handles the rounding; F.interpolate never fires). Big
    # kernels smooth the 3x/5x nearest-upsample staircases; the last (5x) stage runs at
    # 187.5Hz so its channels are kept lean (320) to bound compute before the 2x decimation.
    "medium_15x2_100mel": SMGDecoder1DConfig(
        initial_channels=640,
        stage_channels=[512, 320],
        stage_kernel_sizes=[7, 11],
        time_upsample_factors=[3, 5],       # 15x up -> 187.5Hz
        time_downsample_factor=2,           # 2x down -> 93.75Hz (net 7.5x)
        n_residual_blocks_per_stage=3,
        pre_upsample_residual_blocks=3,
        pre_upsample_kernel_size=7,
        output_dim=100,
    ),
    "small_snake": SMGDecoder1DConfig(
        activation="snake",
        initial_channels=512,
        stage_channels=[512, 256, 256],
        stage_kernel_sizes=[5, 5, 5],
        time_upsample_factors=[2, 2, 1],  # 4x total
    ),
}


@dataclass
class F0PredictorConfig:
    speaker_embedding_dim: int = 192
    # Width of the SIVE encoder features consumed by the F0 predictor's sive_proj.
    # SMG.from_config wires this from sive_encoder_dim; default tracks current SIVE (256).
    encoder_dim: int = 256
    hidden_dim: int = 256
    n_layers: int = 3
    kernel_size: int = 5
    # Width of the CONTENT features fed to a separate voicing branch. When set, the
    # voicing head is split off the F0 trunk and reads content instead of whatever
    # encoder_dim carries; when None, voicing stays on the shared trunk (legacy
    # "features" path). SMG.from_config wires this from sive_encoder_dim iff
    # f0_predictor_input == "contour". See F0Predictor for why the split exists.
    vuv_encoder_dim: Optional[int] = None


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


F0_PREDICTOR_CONFIGS = {
    "default": F0PredictorConfig(),
    "small_deep": F0PredictorConfig(n_layers=5),
}


@dataclass
class F0ConditioningEmbeddingConfig:
    embedding_dim: int = 64
    n_harmonics: int = 6
    sample_rate: int = 16000
    hop_length: int = 256


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


F0_CONDITIONING_EMBEDDING_CONFIGS = {
    "default": F0ConditioningEmbeddingConfig(),
    "small": F0ConditioningEmbeddingConfig(
        n_harmonics=8
    ),
}


@dataclass
class SMGConfig:
    decoder_config: SMGDecoder2DConfig = dataclasses.field(
        default_factory=SMGDecoder2DConfig
    )
    decoder_1d_config: Optional[SMGDecoder1DConfig] = None
    f0_predictor_config: F0PredictorConfig = dataclasses.field(
        default_factory=F0PredictorConfig
    )
    f0_conditioning_embedding_config: F0ConditioningEmbeddingConfig = dataclasses.field(
        default_factory=F0ConditioningEmbeddingConfig
    )

    # reconstruction weighted kl; don't care about latent prior too much
    kl_divergence_loss_weight: float = 1e-6
    free_bits: float = 0.0  # Minimum KL per channel (0 = disabled)

    recon_loss_weight: float = 1.0
    mse_loss_weight: float = 1.0
    l1_loss_weight: float = 1.0

    perceptual_loss_weight: float = 1.0
    multi_scale_mel_weight: float = 1.0

    # What the F0 predictor reads. The two SMG jobs need different answers:
    #
    #   "features"  VOICE CLONING. Content+prosody features already carry the contour;
    #               only the speaker offset changes between speakers, and that is
    #               recoverable from the features plus ECAPA. (This is also why F0
    #               conditioning atrophied on continuous features -- it was redundant,
    #               not broken. The model was right to ignore it.)
    #
    #   "contour"   WORLD-MODEL TRANSLATOR. Input is prosody-free k-means units, so the
    #               contour cannot come from them: predictor(units, speaker) recovers only
    #               ~4% of within-utterance F0 variation over predicting the speaker mean.
    #               Instead the world model supplies a SPEAKER-NORMALIZED contour (it has
    #               the text, where question marks live) and this predictor denormalizes it
    #               using ECAPA -- a far better-posed job. encoder_dim becomes 1.
    #
    # Two purposes, two configs, one codebase.
    f0_predictor_input: str = "features"

    # Discrete-token content input. num_codes>0 makes the SMG accept integer unit ids
    # [B, T'] and embed them via nn.Embedding(num_codes, sive_encoder_dim) instead of
    # continuous features (for pretrained VQ tokenizers like Mimi, which emit ids). The
    # embedding width tracks sive_encoder_dim, so the whole downstream [B, D, T'] float
    # contract is unchanged. code_embed_init controls the table's init/trainability:
    #   "learned_centroid" init from the codebook centroids, trainable (best of both);
    #   "frozen"           init from centroids, requires_grad=False (a fixed prior);
    #   "learned_random"   random init, trainable (strips the tokenizer's geometry).
    # Centroid-init modes need the codebook centroids (passed via --voice_codebook_path).
    num_codes: int = 0
    code_embed_init: str = "learned_centroid"

    f0_loss_weight: float = 5.0
    vuv_loss_weight: float = 2.0


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)

SMG_CONFIGS = {
    "default": SMGConfig(),
    "small": SMGConfig(
        decoder_config=SMG_DECODER_2D_CONFIGS["small"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "small_decoder_only": SMGConfig(
        decoder_config=SMG_DECODER_2D_CONFIGS["small"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "small_1d": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["small"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "small_decoder_only_1d": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["small"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "default_decoder_only_1d": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["default"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "medium_decoder_only_1d_1x": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["medium_1x"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "medium_decoder_only_1d_3x": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["medium_3x"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "medium_decoder_only_1d_3x_f0contourinput": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["medium_3x"],
        f0_predictor_config=F0PredictorConfig(encoder_dim=1, vuv_encoder_dim=256),
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
        f0_predictor_input="contour",
    ),
    "medium_decoder_only_1d_4x": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["medium_4x"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "medium_decoder_only_1d_1x_f0contourinput": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["medium_1x"],
        # encoder_dim / vuv_encoder_dim are both re-derived in SMG.from_config from
        # f0_predictor_input + sive_encoder_dim; set here only so the dataclass reads
        # honestly on its own.
        f0_predictor_config=F0PredictorConfig(encoder_dim=1, vuv_encoder_dim=256),
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
        f0_predictor_input="contour",
    ),
    # Mimi (12.5Hz semantic cb0) -> ids embedded (num_codes=2048, dim=sive_encoder_dim
    # 256) -> 4x time-upsample to mel@50Hz (--voice_hop_length 320, reusing the 50Hz
    # ContentVec mel/vocoder). Contour F0 (Mimi units carry little prosody), voicing on
    # content. Pair with --voice_codebook_path <mimi_semantic_codebook.pt>.
    "medium_decoder_only_1d_4x_mimicontour": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["medium_4x"],
        f0_predictor_config=F0PredictorConfig(encoder_dim=1, vuv_encoder_dim=256),
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
        f0_predictor_input="contour",
        num_codes=2048,
        code_embed_init="learned_centroid",
    ),
    # 24kHz Vocos variant of the above: 100-mel, 8x decoder -> forward resamples to the
    # 93.75Hz Vocos mel target. Pair with --vocoder_config vocos, --voice_hop_length 256,
    # --sample_rate 24000 preprocessing (mel targets from vocos_features.vocos_mel), and
    # --sive_total_stride 7. (The mel/unit rate is 93.75/12.5 = 7.5x, but sive_total_stride
    # is an integer DIVISOR for the collator cap ceil(voice_max_frames/stride); it must round
    # DOWN to 7 so the cap [ceil(937/7)=134] stays >= the stored unit budget [125]. Stride 8
    # under-caps to 118 and truncates ~7 unit frames off the longest 10s clips.)
    "medium_decoder_only_1d_8x_mimicontour_vocos": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["medium_8x_100mel"],
        f0_predictor_config=F0PredictorConfig(encoder_dim=1, vuv_encoder_dim=256),
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
        f0_predictor_input="contour",
        num_codes=2048,
        code_embed_init="learned_centroid",
    ),
    # Same as the 8x Vocos variant but with the EXACT-RATE 15x/2x decoder (net 7.5x) so the
    # output is natively 93.75Hz = mel_length +-1 — no fractional F.interpolate on the mel,
    # which kills the ~6Hz warble + frame-blend slurring the 8x->interpolate path caused.
    "medium_decoder_only_1d_15x2_mimicontour_vocos": SMGConfig(
        decoder_1d_config=SMG_DECODER_1D_CONFIGS["medium_15x2_100mel"],
        f0_predictor_config=F0PredictorConfig(encoder_dim=1, vuv_encoder_dim=256),
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
        f0_predictor_input="contour",
        num_codes=2048,
        code_embed_init="learned_centroid",
    ),
}
