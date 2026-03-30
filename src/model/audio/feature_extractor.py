import torch
import torch.nn as nn

from config.audio.feature_extractor import AUDIO_PRELUDE_CONFIGS, AudioVAEPreludeFeatureExtractorConfig
from model.norms import create_norm
from model.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from model.transformer import MegaTransformerEncoderBlock
from utils import megatransformer_utils
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

        if config.use_input_norm:
            self.input_norm = create_norm(config.feature_channels, config.input_norm_type, config.norm_epsilon)

        self.prelude = nn.ModuleList([
            MegaTransformerEncoderBlock(prelude_config)
            for _ in range(config.n_layers)
        ])

        self.projection = nn.Linear(config.feature_channels, prelude_config.d_model)

        max_mel_timesteps = int(
            (config.sample_rate * config.max_audio_duration) / config.hop_length
        )
        max_sive_timesteps = max_mel_timesteps // config.sive_temporal_stride

        if not prelude_config.use_rotary_embedding:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=prelude_config.d_model,
                max_len=max_sive_timesteps * 2 + 1,
                dropout=0.0
            )

        if config.use_output_norm:
            self.output_norm = create_norm(prelude_config.d_model, config.output_norm_type, config.norm_epsilon)

        self._init_weights()

    def _init_weights(self):
        self.projection.apply(transformer_weight_init())
        for block in self.prelude:
            block.apply(transformer_weight_init())

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "AudioVAEPreludeFeatureExtractor":
        if config_name not in AUDIO_PRELUDE_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(AUDIO_PRELUDE_CONFIGS.keys())}")

        config = AUDIO_PRELUDE_CONFIGS[config_name]
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = AudioVAEPreludeFeatureExtractorConfig(**config_dict)

        return cls(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)

        projected = self.projection(x)  # (batch, T, d_model)

        # megatransformer_utils.print_debug_tensor("embedding audio prelude output", x)

        if hasattr(self, 'pos_encoding'):
            projected = self.pos_encoding(projected)

            # megatransformer_utils.print_debug_tensor("positional encoding audio prelude output", projected)

        x = projected
        for block in self.prelude:
            hidden, _ = block(x)
            x = x + hidden

        # megatransformer_utils.print_debug_tensor("prelude block audio prelude output", x)

        if hasattr(self, 'output_norm'):
            x = self.output_norm(x)

            # megatransformer_utils.print_debug_tensor("normed audio prelude output", x)

        return x
