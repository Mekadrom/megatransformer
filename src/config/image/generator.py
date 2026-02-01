import dataclasses
import json


from dataclasses import dataclass

from config.common import ImageConfig, MegaTransformerBlockConfig


@dataclass
class ImageCodaAndVAEConfig:
    coda_config: MegaTransformerBlockConfig = dataclasses.field(
        default_factory=MegaTransformerBlockConfig
    )
    image_config: ImageConfig = dataclasses.field(
        default_factory=ImageConfig
    )


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


IMAGE_CODA_CONFIGS = {
    "default": ImageCodaAndVAEConfig(),
}
