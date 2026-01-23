from dataclasses import dataclass
import dataclasses
import json

from config.common import AudioConfig, MegaTransformerBlockConfig


@dataclass
class AudioCodaAndVAEConfig:
    coda_config: MegaTransformerBlockConfig = dataclasses.field(
        default_factory=MegaTransformerBlockConfig
    )
    audio_config: AudioConfig = dataclasses.field(
        default_factory=AudioConfig
    )


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
