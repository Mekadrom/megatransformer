import dataclasses
import json

from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioVAEEncoderConfig:
    """Configuration for audio VAE model."""

    # architecture parameters
    encoder_dim: int = 128  # sive feature dim
    latent_dim: int = 16

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
    latent_dim: int = 16

    activation: str = "silu"

    intermediate_channels: Optional[list[int]] = None
    kernel_sizes: Optional[list[int]] = None
    strides: Optional[list[int]] = None
    n_residual_blocks: int = 1
    speaker_embedding_dim: int = 192
    speaker_embedding_proj_dim: Optional[int] = None
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
}
