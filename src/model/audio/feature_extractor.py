import math

import torch
import torch.nn as nn

from config.audio.feature_extractor import AUDIO_PRELUDE_CONFIGS, AudioVAEPreludeFeatureExtractorConfig
from model.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from model.transformer import MegaTransformerBlock
from utils.megatransformer_utils import transformer_weight_init


class AudioVAEPreludeFeatureExtractor(nn.Module):
    """
    Encodes SIVE audio features into token embeddings for the multimodal world model.

    Accepts precomputed SIVE features of shape (batch, feature_channels, timesteps).
    The pipeline:
    1. Transpose to (batch, timesteps, feature_channels)
    2. Linear projection from feature_channels to d_model
    3. 1D positional encoding
    4. Audio-specific prelude transformer with residual connection
    """

    def __init__(self, config: AudioVAEPreludeFeatureExtractorConfig):
        super().__init__()

        self.config = config
        prelude_config = config.prelude_config

        # Normalize raw SIVE features to zero-mean unit-variance before projection.
        # SIVE features can have large range (e.g. [-50, 50]); normalizing puts them
        # on a similar scale to text embeddings for stable interleaving.
        self.input_norm = nn.LayerNorm(config.feature_channels, elementwise_affine=False)

        self.prelude = MegaTransformerBlock(prelude_config)

        self.projection = nn.Linear(config.feature_channels, prelude_config.d_model)

        max_mel_timesteps = int(
            (config.sample_rate * config.max_audio_duration) / config.hop_length
        )
        max_sive_timesteps = max_mel_timesteps // config.sive_temporal_stride

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=prelude_config.d_model,
            max_len=max_sive_timesteps * 2 + 1,
            dropout=0.0
        )

        self._init_weights()

    def _init_weights(self):
        self.projection.apply(transformer_weight_init())
        self.prelude.apply(transformer_weight_init())

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "AudioVAEPreludeFeatureExtractor":
        if config_name not in AUDIO_PRELUDE_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(AUDIO_PRELUDE_CONFIGS.keys())}")

        config = AUDIO_PRELUDE_CONFIGS[config_name]
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = AudioVAEPreludeFeatureExtractorConfig(**config_dict)

        return cls(config)

    def forward(self, x: torch.Tensor, precomputed_latents: bool = True) -> torch.Tensor:
        """
        Process SIVE features into token embeddings.

        Args:
            x: SIVE features of shape (batch, feature_channels, timesteps).
            precomputed_latents: Ignored (kept for interface compatibility).
                SIVE features are always precomputed.

        Returns:
            Token embeddings of shape (batch, timesteps, d_model).
        """
        # x: (batch, C, T) -> (batch, T, C)
        x = x.permute(0, 2, 1).contiguous()

        x = self.input_norm(x)  # normalize per-timestep across channels

        projected = self.projection(x)  # (batch, T, d_model)
        projected = self.pos_encoding(projected)

        prelude_hidden, _ = self.prelude(projected)  # (batch, T, d_model)
        output = projected + prelude_hidden

        return output
