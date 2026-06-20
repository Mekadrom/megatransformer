import dataclasses
import json


from dataclasses import dataclass

from megatransformer.config.common import MegaTransformerBlockConfig


@dataclass
class TextPreludeFeatureExtractorConfig:
    prelude_config: MegaTransformerBlockConfig = dataclasses.field(
        default_factory=MegaTransformerBlockConfig
    )
    n_layers: int = 1

    d_model: int = 512
    vocab_size: int = 32009
    hidden_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    use_pos_emb_ovr: bool = False

    use_output_norm: bool = False
    norm_epsilon: float = 1e-5
    output_norm_type: str = "layernorm"

    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


TEXT_FEATURE_EXTRACTOR_CONFIGS = {
    "default": TextPreludeFeatureExtractorConfig(),
}
