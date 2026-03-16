import torch
import torch.nn as nn

from typing import Optional

from config.audio.generator import AUDIO_CODA_CONFIGS, AudioCodaAndVAEConfig
from model.transformer import MegaTransformerBlock
from utils.megatransformer_utils import transformer_weight_init


class TemporalRefine(nn.Module):
    """Temporal refinement via Conv1d for SIVE feature predictions.

    Two Conv1d(kernel_size=3) layers with GELU activation give a 5-frame
    receptive field (~240ms at 16kHz / hop=256 / SIVE 3x downsample),
    roughly one phoneme — enough for local coherence without smearing
    across phoneme boundaries.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, timesteps)"""
        return x + self.conv2(self.act(self.conv1(x)))


class AudioCodaAndVAEWithLoss(nn.Module):
    """
    Audio/voice output head for the multimodal world model.

    Predicts SIVE features from transformer hidden states.
    Output shape: (batch, feature_channels, timesteps).

    The pipeline:
    1. Coda transformer with residual connection
    2. Linear projection from d_model to feature_channels
    3. Transpose to (batch, feature_channels, timesteps)
    4. (Optional) Conv1d temporal refinement

    Works for both general audio and voice — just use separate instances.
    Decoding SIVE features to mel/waveform is handled externally (CVAE + vocoder).
    """

    def __init__(self, prefix: str, config: AudioCodaAndVAEConfig):
        super(AudioCodaAndVAEWithLoss, self).__init__()

        self.prefix = prefix
        self.config = config
        self.feature_channels = config.feature_channels

        coda_config = config.coda_config
        self.coda = nn.ModuleList([
            MegaTransformerBlock(coda_config)
            for _ in range(config.n_layers)
        ])

        self.feature_projection = nn.Linear(coda_config.d_model, config.feature_channels)

        # Optional temporal refinement after linear projection
        output_mode = getattr(config, 'output_mode', 'linear')
        self.temporal_refine = TemporalRefine(config.feature_channels) if output_mode == "conv_refine" else None

        # Learnable denormalization: maps from normalized space back to original
        # latent distribution. Initialized to identity (scale=1, bias=0).
        self.output_scale = nn.Parameter(torch.ones(config.feature_channels))
        self.output_bias = nn.Parameter(torch.zeros(config.feature_channels))

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self._init_weights()

    def _init_weights(self):
        for block in self.coda:
            block.apply(transformer_weight_init())
        self.feature_projection.apply(transformer_weight_init())
        if self.temporal_refine is not None:
            self.temporal_refine.apply(transformer_weight_init())

    @classmethod
    def from_config(cls, prefix: str, config_name: str, **overrides) -> "AudioCodaAndVAEWithLoss":
        if config_name not in AUDIO_CODA_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(AUDIO_CODA_CONFIGS.keys())}")

        config = AUDIO_CODA_CONFIGS[config_name]
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = AudioCodaAndVAEConfig(**config_dict)

        return cls(prefix, config)

    def forward(
        self,
        x: torch.Tensor,
        latent_labels: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        decode_to_mel: bool = False,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Process hidden states through coda to predict SIVE features.

        Args:
            x: Input tensor of shape (batch, seq_length, d_model).
            latent_labels: Target SIVE features for loss computation.
                Shape: (batch, feature_channels, timesteps)
            lengths: Unused (kept for interface compatibility).
            decode_to_mel: Unused (kept for interface compatibility).

        Returns:
            Dictionary containing:
            - "{prefix}_latent_preds": Predicted SIVE features (batch, feature_channels, timesteps)
            - "{prefix}_latent_l1_loss", "{prefix}_latent_mse_loss": Losses (if latent_labels provided)
        """
        h = x
        for block in self.coda:
            hidden, _ = block(h)
            h = h + hidden

        feature_preds = self.feature_projection(h)  # (batch, seq_length, feature_channels)
        # Denormalize: learnable scale and bias map back to original latent range
        feature_preds = feature_preds * self.output_scale + self.output_bias
        feature_preds = feature_preds.permute(0, 2, 1)  # (batch, feature_channels, timesteps)

        if self.temporal_refine is not None:
            feature_preds = self.temporal_refine(feature_preds)

        outputs = {f"{self.prefix}_latent_preds": feature_preds}

        if latent_labels is not None:
            outputs[f"{self.prefix}_latent_l1_loss"] = self.l1_loss(feature_preds, latent_labels)
            outputs[f"{self.prefix}_latent_mse_loss"] = self.mse_loss(feature_preds, latent_labels)

        return outputs
