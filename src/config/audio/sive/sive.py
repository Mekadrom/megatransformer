import dataclasses
import json

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpeakerInvariantVoiceEncoderConfig:
    """Configuration for SIVE model."""
    audio_n_mels: int = 80
    encoder_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    conv_kernel_sizes: Optional[list[int]] = None
    conv_strides: Optional[list[int]] = None
    vocab_size: int = 30  # " 'abcdefghijklmnopqrstuvwxyz" + blank + UNKOWN
    num_speakers: int = 921
    dropout: float = 0.1
    max_seq_len: int = 4096
    speaker_pooling: str = "mean"

    # Dropout regularization (helps prevent memorization)
    conv_dropout: float = 0.0         # Dropout1d in conv frontend (0 = use standard dropout)
    feature_dropout: float = 0.0      # Dropout on features before heads
    head_dropout: float = 0.0         # Dropout in ASR head
    attention_head_drop: float = 0.0  # DropHead on attention

    # Architectural options
    conformer_kernel_size: int = 31  # Kernel size for Conformer conv (standard from paper)
    activation: str = "swiglu"

    # SpecAugment (data augmentation for ASR)
    use_spec_augment: bool = False
    spec_time_mask_param: int = 50   # Max time mask width (T)
    spec_freq_mask_param: int = 20   # Max frequency mask width (F)
    spec_num_time_masks: int = 2     # Number of time masks
    spec_num_freq_masks: int = 2     # Number of frequency masks

    # Stochastic Depth (drop entire residual paths)
    drop_path_rate: float = 0.0  # Max drop rate (linearly scaled per layer, 0=disabled)

    # Variance regularization (for VAE-friendly features)
    use_variance_reg: bool = False
    temporal_var_weight: float = 0.01
    temporal_var_min: float = 0.1
    dim_var_weight: float = 0.01
    dim_var_min: float = 0.1
    temporal_smoothness_weight: float = 0.1
    temporal_smoothness_max: float = 0.95

    # CTC upsampling (relaxes CTC length constraint without increasing transformer cost)
    # Upsamples features before CTC head using linear interpolation
    # factor=1: no upsampling (default), factor=2: 2x more CTC frames, etc.
    ctc_upsample_factor: int = 1

    speaker_classifier_hidden_dim: Optional[int] = None

    def __post_init__(self):
        # defaults
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [7, 3, 3]  # Larger first kernel for more acoustic context
        if self.conv_strides is None:
            self.conv_strides = [2, 2, 1]  # 4x downsampling

        if self.speaker_classifier_hidden_dim is None:
            self.speaker_classifier_hidden_dim = self.encoder_dim * 2

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


CONFIGS = {
    "default": SpeakerInvariantVoiceEncoderConfig(),
    # ~1.3M params - very fast, good for experimentation
    "tiny": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=128,
        num_layers=3,
        num_heads=4,
        ff_dim=512,
        dropout=0.1,
    ),
    # ~7M params w/ macaron and swiglu, ~2.5M w/o
    # This is the current best model with the run name "tiny_deep_0_4" - checkpoint-60000 was best performance, least overfit.
    "tiny_deep": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=128,
        num_layers=12,
        num_heads=8,
        ff_dim=512,
        dropout=0.1,
        ctc_upsample_factor=2,
    ),
    "tiny_deep_2xdownsample": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=128,
        num_layers=12,
        num_heads=8,
        ff_dim=512,
        dropout=0.1,
        conv_strides=[2, 1, 1],  # 2x downsampling instead of 4x (more CTC frames, higher transformer cost)
        ctc_upsample_factor=1,
    ),
    "tiny_deep_4xupsample": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=128,
        num_layers=12,
        num_heads=8,
        ff_dim=512,
        dropout=0.1,
        ctc_upsample_factor=4,
    ),
    # ~9.6M params w/ macaron and swiglu, ~3.7M w/o
    "small": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=256,
        num_layers=4,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1,
    ),
    # ~30M params w/ macaron and swiglu, ~11M w/o
    "small_deep": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=256,
        num_layers=12,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
    ),
    # ~32M params w/ macaron and swiglu, ~11M w/o
    "medium": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=384,
        num_layers=6,
        num_heads=6,
        ff_dim=1536,
        dropout=0.1,
    ),
    # ~61M params w/ macaron and swiglu, ~23M w/o
    "medium_deep": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=384,
        num_layers=12,
        num_heads=6,
        ff_dim=1536,
        dropout=0.1,
    ),
    # ~74M params w/ macaron and swiglu, ~27M w/o
    "large": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=512,
        num_layers=8,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
    ),
    # 110M params w/ macaron and swiglu, ~40M w/o
    "large_deep": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
    ),
}
