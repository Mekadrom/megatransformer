from dataclasses import dataclass
import dataclasses
import json

from typing import Optional

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

    # Discrete-unit output. When set to a codebook size K, the coda gains a K-way
    # classifier head alongside the continuous regression head, and the trainer can
    # supervise it with cross-entropy against k-means unit ids.
    #
    # Why this exists: trained to REGRESS continuous content frames, the model predicts
    # frame t from the audio history alone -- L1/NLL only ask it to be approximately
    # right, and "approximately right" is reachable by extrapolating acoustics, so the
    # text earns no gradient and gets ignored (measured: cosine-sim to target ~0.02,
    # text-embedding grad ~7e-4, fluent babble). Cross-entropy over units cannot be
    # satisfied that way: the model has to NAME the next unit, and naming it requires
    # the text. This is what VALL-E / SPEAR-TTS / AudioLM buy with discrete tokens.
    #
    # None = off; the coda is the continuous regressor it has always been.
    unit_vocab_size: Optional[int] = None

    # F0/VUV regression head, beside the unit classifier.
    #
    # Quantization strips prosody from the content units by construction, so the contour
    # has to come from somewhere. The SMG's own F0 predictor can only see (units, speaker)
    # — measured, it recovers ~4% of within-utterance F0 variation over just predicting the
    # speaker's mean pitch (r=0.45, right shape, squashed amplitude). It cannot do better,
    # because neither units nor an ECAPA embedding knows the sentence is a question.
    #
    # The world coda does know: it sits on text conditioning. So the split is units for
    # content (cross-entropy — forces naming, forces reading the text) and F0 for prosody
    # (regression — a contour is genuinely continuous). The SMG's predictor remains the
    # fallback for standalone use where no external contour exists.
    predict_f0: bool = False
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
