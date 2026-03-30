import dataclasses
import json


from dataclasses import dataclass

from config.common import ImageConfig, MegaTransformerBlockConfig


@dataclass
class ImageVAEPreludeFeatureExtractorConfig:
    prelude_config: MegaTransformerBlockConfig = dataclasses.field(
        default_factory=MegaTransformerBlockConfig
    )
    n_layers: int = 1
    image_config: ImageConfig = dataclasses.field(
        default_factory=ImageConfig
    )

    use_input_norm: bool = False
    use_output_norm: bool = False
    norm_type: str = "layernorm"
    norm_epsilon: float = 1e-5
    input_norm_type: str = "instancenorm"  # "instancenorm" (per channel, across spatial) or "layernorm" (across channels)
    output_norm_type: str = "instancenorm"  # "instancenorm" (per channel, across spatial) or "layernorm" (across channels)


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


IMAGE_PRELUDE_CONFIGS = {
    "default": ImageVAEPreludeFeatureExtractorConfig(),
}
