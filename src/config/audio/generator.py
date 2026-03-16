from dataclasses import dataclass
import dataclasses
import json

from config.common import MegaTransformerBlockConfig


@dataclass
class AudioCodaAndVAEConfig:
    """Config for the audio coda that predicts SIVE features.

    The coda projects d_model hidden states back to feature_channels,
    producing SIVE-shaped outputs (B, feature_channels, T).
    """
    coda_config: MegaTransformerBlockConfig = dataclasses.field(
        default_factory=MegaTransformerBlockConfig
    )
    n_layers: int = 1
    feature_channels: int = 128
    # "linear" (single linear, original) or "conv_refine" (linear + Conv1d refinement)
    output_mode: str = "linear"


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


AUDIO_CODA_CONFIGS = {
    "default": AudioCodaAndVAEConfig(),
}
