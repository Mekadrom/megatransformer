import dataclasses
import json


from dataclasses import dataclass

from config.common import MegaTransformerBlockConfig



@dataclass
class TextCodaClassifierConfig:
    coda_config: MegaTransformerBlockConfig = dataclasses.field(
        default_factory=MegaTransformerBlockConfig
    )
    vocab_size: int = 32007
    label_smoothing: float = 0.0


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


TEXT_CODA_CONFIGS = {
    "default": TextCodaClassifierConfig(),
}
