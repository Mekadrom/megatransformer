import dataclasses
import json

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class SpeakerInvariantVoiceEncoderConfig:
    """Configuration for SIVE model."""
    voice_n_mels: int = 80
    encoder_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    use_conv2d_frontend: bool = False
    conv_kernel_sizes: Optional[list[Union[int, tuple[int, int], list[int]]]] = None
    conv_strides: Optional[list[Union[int, tuple[int, int], list[int]]]] = None
    vocab_size: int = 30  # " 'abcdefghijklmnopqrstuvwxyz" + blank + UNKOWN
    num_speakers: int = 2338
    dropout: float = 0.1
    max_seq_len: int = 4096
    speaker_pooling: str = "mean"

    # Dropout regularization (helps prevent memorization)
    conv_dropout: float = 0.0         # Dropout1d in conv frontend (0 = use standard dropout)
    feature_dropout: float = 0.0      # Dropout on features before heads
    head_dropout: float = 0.0         # Dropout in ASR head
    attention_head_drop: float = 0.0  # DropHead on attention

    # Architectural options
    conformer_kernel_size: int = 31  # Kernel size for Conformer conv (standard from paper)
    activation: str = "swiglu"

    # SpecAugment (data augmentation for ASR)
    use_spec_augment: bool = False
    spec_time_mask_param: int = 50   # Max time mask width (T)
    spec_freq_mask_param: int = 20   # Max frequency mask width (F)
    spec_num_time_masks: int = 2     # Number of time masks
    spec_num_freq_masks: int = 2     # Number of frequency masks

    # Mel-space noise injection (robustness augmentation; alternative to
    # waveform-level noise addition that doesn't require storing waveforms).
    use_mel_noise: bool = False
    mel_noise_snr_min_db: float = 5.0    # lower bound on sampled target SNR
    mel_noise_snr_max_db: float = 20.0   # upper bound on sampled target SNR
    mel_noise_prob: float = 0.5          # per-utterance application probability

    # Mel-space frequency-response modulation (simulates mic/channel EQ
    # variation). Multiplies each mel band by a random smooth gain curve.
    use_mel_freq_response: bool = False
    mel_freq_response_strength: float = 0.3  # std of pre-smoothing gain noise
    mel_freq_response_prob: float = 0.5      # per-utterance application probability
    mel_freq_response_smoothing: int = 7     # smoothing kernel width (odd; larger = smoother EQ)

    # Post-hoc VTLP (Vocal Tract Length Perturbation). Piecewise-linear warp of
    # the mel-bin axis. Approximate vs filter-bank-level VTLP, but cheap and a
    # useful regularizer alongside waveform-level pitch shift.
    use_mel_vtlp: bool = False
    mel_vtlp_strength: float = 0.1       # alpha drawn from [1-strength, 1+strength]
    mel_vtlp_prob: float = 0.5           # per-utterance application probability
    mel_vtlp_boundary_frac: float = 0.7  # fraction of mel-bin axis under linear region

    # Stochastic Depth (drop entire residual paths)
    drop_path_rate: float = 0.0  # Max drop rate (linearly scaled per layer, 0=disabled)

    # Variance regularization (for VAE-friendly features) — legacy var-hinge.
    # Currently unused by the trainer (no consumer of result["variance_loss"]).
    # Kept for backwards compat. New runs should prefer use_std_hinge below.
    use_variance_reg: bool = False
    temporal_var_weight: float = 0.01
    temporal_var_min: float = 0.1
    dim_var_weight: float = 0.01
    dim_var_min: float = 0.1
    temporal_smoothness_weight: float = 0.1
    temporal_smoothness_max: float = 0.95

    # Std-based hinge on per-dim std (disabled by default). Replaces the
    # var-hinge with constant gradient pressure as a dim approaches zero std.
    # Independently togglable from covariance reg below.
    use_std_hinge: bool = False
    dim_std_min: float = 0.5      # target minimum per-dim std
    dim_std_weight: float = 1.0   # weight on dim-std hinge loss
    # Temporal std hinge: penalizes too-flat frame-to-frame deltas. Set
    # temporal_std_weight > 0 to enable (orthogonal axis from dim hinge).
    temporal_std_min: float = 0.1
    temporal_std_weight: float = 0.0

    # VICReg-style covariance / decorrelation regularization (disabled by
    # default). Penalizes off-diagonal of the feature covariance matrix.
    use_covariance_reg: bool = False
    cov_weight: float = 0.04      # VICReg paper default

    # CTC upsampling (relaxes CTC length constraint without increasing transformer cost)
    # Upsamples features before CTC head using linear interpolation
    # factor=1: no upsampling (default), factor=2: 2x more CTC frames, etc.
    ctc_upsample_factor: int = 1

    # Norm applied inside the conv subsampling frontend (per conv layer). LIVE:
    # passed through to Conv2dSubsampling/ConvSubsampling. One of "batchnorm",
    # "instancenorm", "groupnorm", "layernorm", "rmsnorm", "none", or None
    # (-> instancenorm). Distinct from final_norm_type (post-encoder output norm).
    downsample_norm_type: Optional[str] = None

    # SIVE has four independent norm levers. (1) downsample_norm_type = conv
    # frontend; (2) final_norm_type = post-encoder output; plus the two below.
    # block_norm_type (lever 3) = the transformer encoder pre-norms (macaron
    # FFNs, attention, and the conformer module's input pre-norm), over the
    # feature dim of [B, T, D]. One of "layernorm" (default), "rmsnorm", "none".
    block_norm_type: str = "layernorm"
    # conv_norm_type (lever 4) = the norm on the conformer depthwise-conv output
    # [B, C, T]. One of "instancenorm" (default, matches prior behavior),
    # "batchnorm", "groupnorm", "layernorm", "rmsnorm", "none".
    conv_norm_type: str = "instancenorm"

    # Final normalization on encoder features (the user-facing SIVE output).
    # "layernorm" (default, matches prior behavior), "rmsnorm" (no mean-subtraction
    # → less dim-axis competition; can help when a few dims appear blown out and
    # invariant), or "none" (skip — let consumers normalize).
    final_norm_type: str = "layernorm"

    speaker_classifier_hidden_dim: Optional[int] = None

    # GRL speaker-adversary target: "speaker_id" (cross-entropy over training
    # speakers) or "ecapa_embedding" (cosine-regress the ECAPA embedding — richer,
    # generalizes to unseen speakers, and enforces features orthogonal to the
    # SMG's speaker vector). Embedding mode sizes the head to speaker_embedding_dim.
    speaker_adversary_target: str = "speaker_id"
    speaker_embedding_dim: int = 192
    # Heads for the multi-head poolings ("mhasp", "multi_head_attention").
    speaker_classifier_num_heads: int = 4
    # Encoder layer the GRL speaker adversary attaches to. -1 = final layer (default,
    # current behavior: GRL + CTC both on the output). >=0 reverses into all_hiddens[grl_layer]
    # (0 = conv frontend, 1..N = blocks) while CTC stays on the final layer — aligns the
    # clean-point with the SMG's tap (layer 10) so the layer the SMG consumes is the scrubbed one.
    grl_layer: int = -1

    # Gender GRL adversary — a binary (male/female) pooled head reversed into its
    # OWN tap (gender_grl_layer, default 10), independent of the speaker adversary's
    # grl_layer. Off by default. Targets the gender direction the speaker GRL leaves
    # largely intact — gender leaks ~0.91 balanced-acc in every run measured because
    # nothing removes it today. gender_ids come from the shards (0=male, 1=female,
    # -1=unknown → ignored by the CE). The loss weight is a trainer arg
    # (--gender_grl_weight), mirroring how grl_weight lives there.
    use_gender_grl: bool = False
    num_genders: int = 2
    gender_pooling: Optional[str] = None  # None -> mirror speaker_pooling
    # Encoder layer the gender adversary reverses into — independent of the speaker
    # adversary's grl_layer. Default 10 = the SMG's tap point (scrub gender where the
    # SMG reads). -1 = final layer; 0 = conv frontend; 1..N = block outputs. Only
    # validated / used when use_gender_grl is set.
    gender_grl_layer: int = 10

    def __post_init__(self):
        # defaults
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [7, 3, 3]  # Larger first kernel for more acoustic context
        if self.conv_strides is None:
            self.conv_strides = [2, 2, 1]  # 4x downsampling
        if self.downsample_norm_type is None:
            self.downsample_norm_type = "instancenorm"

        if self.speaker_classifier_hidden_dim is None:
            self.speaker_classifier_hidden_dim = self.encoder_dim * 2

        if self.gender_pooling is None:
            self.gender_pooling = self.speaker_pooling

        if self.grl_layer != -1 and not (0 <= self.grl_layer <= self.num_layers):
            raise ValueError(f"grl_layer={self.grl_layer} out of range: use -1 (final) or 0..{self.num_layers}")

        # Only validate the gender tap when the head is active — the default (10)
        # would otherwise trip a small-config default construction (num_layers=4).
        if self.use_gender_grl and self.gender_grl_layer != -1 and not (0 <= self.gender_grl_layer <= self.num_layers):
            raise ValueError(
                f"gender_grl_layer={self.gender_grl_layer} out of range: use -1 (final) or 0..{self.num_layers}"
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        return json.dumps(self.to_dict(), indent=2)


CONFIGS = {
    # The single SIVE base: 256-dim, 12L, conv2d 3x-downsample, attentive pooling.
    # All four norm levers sit at their defaults and are overridable per-run from
    # the CLI (--downsample_norm_type / --block_norm_type / --conv_norm_type /
    # --final_norm_type), so norm ablations need no separate presets:
    #   frontend (downsample) = instancenorm  (explicit below; the name has no norm token)
    #   block pre-norms       = layernorm     (dataclass default)
    #   conformer conv        = instancenorm  (dataclass default)
    #   final output norm     = layernorm     (dataclass default)
    # Older size/norm presets were pruned 2026-06-29; recover any from git history
    # or a run's tagged commit if a structural variant (e.g. 128-dim) is needed.
    "small_deep_3xdownsample_conv2d_attentive": SpeakerInvariantVoiceEncoderConfig(
        encoder_dim=256,
        num_layers=12,
        num_heads=8,
        ff_dim=512,
        dropout=0.1,
        use_conv2d_frontend=True,
        conv_kernel_sizes=[(5, 7), (5, 3), (5, 3)],
        conv_strides=[(2, 3), (2, 1), (1, 1)],
        ctc_upsample_factor=2,
        downsample_norm_type="instancenorm",
        speaker_pooling="attentive_statistics",
    ),
}
