import dataclasses
import json


from dataclasses import dataclass


@dataclass
class TextFeatureExtractorConfig:
    d_model: int = 512
    vocab_size: int = 32007
    norm_type: str = "layernorm"
    layer_norm_epsilon: float = 1e-5
    hidden_dropout_prob: float = 0.1


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


TEXT_FEATURE_EXTRACTOR_CONFIGS = {
    "default": TextFeatureExtractorConfig(),
}
