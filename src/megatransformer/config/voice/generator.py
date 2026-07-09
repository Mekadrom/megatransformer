from dataclasses import dataclass
import dataclasses
import json

from megatransformer.config.common import MegaTransformerBlockConfig


@dataclass
class VoiceCodaAndSMGConfig:
    """Config for the voice coda that predicts SIVE features.

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

    use_input_norm: bool = False
    use_output_norm: bool = False
    norm_epsilon: float = 1e-5
    input_norm_type: str = "layernorm"
    output_norm_type: str = "layernorm"

    # Stochastic output head. When True, the coda additionally predicts a
    # per-frame, per-dim log-variance (a parallel linear head off the coda
    # hidden state) so the SIVE-feature output is a heteroscedastic diagonal
    # Gaussian rather than a point estimate. Training uses (beta-)NLL; inference
    # samples mu + temperature*std*eps. Default False = original deterministic
    # behaviour (byte-identical: the extra head is simply not built).
    stochastic_output: bool = False
    logvar_clamp_min: float = -8.0
    logvar_clamp_max: float = 4.0
    # Initial (homoscedastic) log-variance: the log-variance head's weight inits
    # to zero and its bias to this, so the model starts by predicting a constant
    # variance everywhere and learns position-conditional variance from there.
    logvar_init: float = 0.0


    def __post_init__(self):
        pass

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


VOICE_CODA_CONFIGS = {
    "default": VoiceCodaAndSMGConfig(),
}
