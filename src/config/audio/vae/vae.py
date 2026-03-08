import dataclasses
import json

from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioVAEEncoderConfig:
    """Configuration for audio VAE model."""

    # architecture parameters
    encoder_dim: int = 128  # sive feature dim
    latent_channels: int = 16

    activation: str = "silu"

    intermediate_channels: Optional[list[int]] = None
    kernel_sizes: Optional[list[int]] = None
    strides: Optional[list[int]] = None
    n_residual_blocks: int = 1

    # training parameters
    dropout: float = 0.1
    logvar_clamp_max: float = 4.0


    def __post_init__(self):
        # defaults
        if self.intermediate_channels is None:
            self.intermediate_channels = [256, 256, 128]
        if self.kernel_sizes is None:
            self.kernel_sizes = [5, 5, 3]
        if self.strides is None:
            self.strides = [2, 2, 2]  # 8x total downsampling
            
    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


AUDIO_ENCODER_CONFIGS = {
    "default": AudioVAEEncoderConfig(),
    "small": AudioVAEEncoderConfig(
        intermediate_channels=[256],
        kernel_sizes=[3],
        strides=[1],  # no downsampling
        n_residual_blocks=4,
    ),
}


@dataclass
class AudioVAEDecoderConfig:
    """Configuration for audio VAE decoder model."""

    # architecture parameters
    latent_channels: int = 16

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


AUDIO_DECODER_CONFIGS = {
    "default": AudioVAEDecoderConfig(),
    "small": AudioVAEDecoderConfig(
        intermediate_channels_2d=[64, 96, 144],
        kernel_sizes_2d=[(3, 5), (3, 5), (3, 5)],
        scale_factors_2d=[(2, 2), (2, 2), (2, 1)],  # 8x freq, 4x time
    ),
}


@dataclass
class AudioVAEDecoder1DConfig:
    """Configuration for Conv1D-only audio VAE decoder.

    Treats frequency as channels throughout — no 1D→2D reshape.
    Avoids ring artifacts caused by Conv2D spatial isotropy assumption.
    """

    latent_channels: int = 16
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


AUDIO_DECODER_1D_CONFIGS = {
    "default": AudioVAEDecoder1DConfig(),
    "small": AudioVAEDecoder1DConfig(
        initial_channels=512,
        stage_channels=[512, 256, 256],
        stage_kernel_sizes=[5, 5, 5],
        time_upsample_factors=[2, 2, 1],  # 4x total
    ),
    "medium": AudioVAEDecoder1DConfig(
        initial_channels=640,
        stage_channels=[640, 384, 256],
        stage_kernel_sizes=[7, 5, 5],
        time_upsample_factors=[2, 2, 1],  # 4x total
        n_residual_blocks_per_stage=3,
        pre_upsample_residual_blocks=3,
        pre_upsample_kernel_size=7,
    ),
    "medium_3x": AudioVAEDecoder1DConfig(
        initial_channels=640,
        stage_channels=[640, 384, 256],
        stage_kernel_sizes=[7, 5, 5],
        time_upsample_factors=[3, 1, 1],  # 3x total
        n_residual_blocks_per_stage=3,
        pre_upsample_residual_blocks=3,
        pre_upsample_kernel_size=7,
    ),
    "small_snake": AudioVAEDecoder1DConfig(
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
    encoder_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 3
    kernel_size: int = 5


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
class AudioVAEConfig:
    encoder_config: AudioVAEEncoderConfig = dataclasses.field(
        default_factory=AudioVAEEncoderConfig
    )
    decoder_config: AudioVAEDecoderConfig = dataclasses.field(
        default_factory=AudioVAEDecoderConfig
    )
    decoder_1d_config: Optional[AudioVAEDecoder1DConfig] = None
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
    
AUDIO_VAE_CONFIGS = {
    "default": AudioVAEConfig(),
    "small": AudioVAEConfig(
        encoder_config=AUDIO_ENCODER_CONFIGS["small"],
        decoder_config=AUDIO_DECODER_CONFIGS["small"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "small_decoder_only": AudioVAEConfig(
        encoder_config=AUDIO_ENCODER_CONFIGS["default"],
        decoder_config=AUDIO_DECODER_CONFIGS["small"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "small_1d": AudioVAEConfig(
        encoder_config=AUDIO_ENCODER_CONFIGS["small"],
        decoder_1d_config=AUDIO_DECODER_1D_CONFIGS["small"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "small_decoder_only_1d": AudioVAEConfig(
        encoder_config=AUDIO_ENCODER_CONFIGS["default"],
        decoder_1d_config=AUDIO_DECODER_1D_CONFIGS["small"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "default_decoder_only_1d": AudioVAEConfig(
        encoder_config=AUDIO_ENCODER_CONFIGS["default"],
        decoder_1d_config=AUDIO_DECODER_1D_CONFIGS["default"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "medium_decoder_only_1d": AudioVAEConfig(
        encoder_config=AUDIO_ENCODER_CONFIGS["default"],
        decoder_1d_config=AUDIO_DECODER_1D_CONFIGS["medium"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "medium_decoder_only_1d_3x": AudioVAEConfig(
        encoder_config=AUDIO_ENCODER_CONFIGS["default"],
        decoder_1d_config=AUDIO_DECODER_1D_CONFIGS["medium_3x"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
    "small_decoder_only_1d_snake": AudioVAEConfig(
        encoder_config=AUDIO_ENCODER_CONFIGS["default"],
        decoder_1d_config=AUDIO_DECODER_1D_CONFIGS["small_snake"],
        f0_predictor_config=F0_PREDICTOR_CONFIGS["default"],
        f0_conditioning_embedding_config=F0_CONDITIONING_EMBEDDING_CONFIGS["small"],
    ),
}
