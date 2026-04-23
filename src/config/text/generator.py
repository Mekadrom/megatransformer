import dataclasses
import json


from dataclasses import dataclass
from typing import Optional

from config.common import MegaTransformerBlockConfig



@dataclass
class TextCodaClassifierConfig:
    coda_config: MegaTransformerBlockConfig = dataclasses.field(
        default_factory=MegaTransformerBlockConfig
    )
    n_layers: int = 1
    vocab_size: int = 32009
    label_smoothing: float = 0.0

    # Soft logit capping on LM head output (Gemma 2-style). Applied before
    # loss computation: logits = cap * tanh(logits / cap). Prevents logit
    # explosion and pairs well with label smoothing.
    # None = disabled. Default 30.0 matches Gemma 2.
    lm_head_logit_cap: Optional[float] = 30.0

    use_input_norm: bool = False
    norm_epsilon: float = 1e-5
    input_norm_type: str = "layernorm"  # "instancenorm" (


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
