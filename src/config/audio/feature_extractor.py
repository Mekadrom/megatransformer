import dataclasses
import json


from dataclasses import dataclass

from config.common import MegaTransformerBlockConfig


@dataclass
class AudioVAEPreludeFeatureExtractorConfig:
    """Config for the audio prelude feature extractor that accepts SIVE features.

    SIVE features are (B, feature_channels, T) — 1D sequences of feature vectors.
    The prelude projects these to d_model, adds positional encoding, and runs
    a small transformer for audio-specific processing before interleaving.
    """
    prelude_config: MegaTransformerBlockConfig = dataclasses.field(
        default_factory=MegaTransformerBlockConfig
    )
    n_layers: int = 1
    feature_channels: int = 128
    sample_rate: int = 16000
    hop_length: int = 256
    max_audio_duration: float = 30.0
    sive_temporal_stride: int = 4


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


AUDIO_PRELUDE_CONFIGS = {
    "default": AudioVAEPreludeFeatureExtractorConfig(
    ),
}
