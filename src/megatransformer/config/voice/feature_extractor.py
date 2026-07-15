import dataclasses
import json


from dataclasses import dataclass

from megatransformer.config.common import MegaTransformerBlockConfig


@dataclass
class VoiceSIVEPreludeFeatureExtractorConfig:
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
    max_audio_duration: float = 10.0
    sive_temporal_stride: int = 4

    # Tacotron-2-style prenet bottleneck on the AUTOREGRESSIVE path (shifted teacher
    # forcing during training, own-output feedback during generation). Under teacher
    # forcing the true previous frames are so informative that the text contributes
    # nothing to the loss, so the text pathway gets no gradient and the model converges
    # to an unconditional speech model — fluent babble, cosine-sim ~0 to the target.
    # Corrupting this path forces the text to become the reliable information source.
    # Applied with training=True ALWAYS (including inference), as Tacotron 2 requires:
    # turning it off at inference restores the over-reliance it exists to prevent.
    # 0.0 = off. 0.5 is the Tacotron 2 value.
    prenet_dropout: float = 0.0

    use_input_norm: bool = False
    use_output_norm: bool = False
    norm_epsilon: float = 1e-5
    input_norm_type: str = "layernorm"
    output_norm_type: str = "layernorm"


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


VOICE_PRELUDE_CONFIGS = {
    "default": VoiceSIVEPreludeFeatureExtractorConfig(),
}
