import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from typing import List, Optional, Tuple, Union

from megatransformer.config.voice.feature_extractor import VOICE_PRELUDE_CONFIGS, VoiceSIVEPreludeFeatureExtractorConfig
from megatransformer.model.norms import create_norm
from megatransformer.model.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from megatransformer.model.transformer import MegaTransformerEncoderBlock
from megatransformer.utils.megatransformer_utils import (
    apply_depth_scaled_residual_init,
    linear_weight_init,
)


class VoiceSIVEPreludeFeatureExtractor(nn.Module):
    """
    Encodes SIVE features into token embeddings for the multimodal world model.

    Accepts precomputed SIVE features of shape (batch, feature_channels, timesteps).
    The pipeline:
    1. Transpose to (batch, timesteps, feature_channels)
    2. Linear projection from feature_channels to d_model
    3. 1D positional encoding
    4. Voice-specific prelude transformer with residual connection
    """

    def __init__(self, config: VoiceSIVEPreludeFeatureExtractorConfig):
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

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        init_linear = linear_weight_init(gain=1.0)
        self.projection.apply(init_linear)
        for block in self.prelude:
            block.apply(init_linear)
        apply_depth_scaled_residual_init(self.prelude)

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "VoiceSIVEPreludeFeatureExtractor":
        if config_name not in VOICE_PRELUDE_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(VOICE_PRELUDE_CONFIGS.keys())}")

        config = VOICE_PRELUDE_CONFIGS[config_name]
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = VoiceSIVEPreludeFeatureExtractorConfig(**config_dict)

        return cls(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: Optional[List] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        apply_prenet_dropout: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Process SIVE features into token embeddings.

        Args:
            x: SIVE features of shape (batch, feature_channels, timesteps).
            kv_caches: Optional list of KVCache objects, one per prelude layer.
                Used for autoregressive voice synthesis at inference time so
                that frame t's prelude self-attention sees frames 0..t-1.
            position_offset: RoPE position offset for cached generation.
            use_cache: If True, return (embeddings, kv_caches) tuple.
            apply_prenet_dropout: Enable the config's prenet_dropout bottleneck. Only for
                the AUTOREGRESSIVE path (shifted teacher forcing / own-output feedback),
                where the previous-frame signal otherwise drowns out the text. Must stay
                False when the prelude is READING real audio (transcription, cross-modal):
                there the audio is the input, not a crutch, and corrupting it is just
                damage.

        Returns:
            Token embeddings of shape (batch, timesteps, d_model).
            If use_cache=True, returns (embeddings, new_kv_caches) tuple.
        """
        # x: (batch, C, T) -> (batch, T, C)
        x = x.permute(0, 2, 1).contiguous()

        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)

        projected = self.projection(x)  # (batch, T, d_model)

        # Tacotron-2 prenet bottleneck. training=True unconditionally: Tacotron 2 keeps
        # this dropout ON at inference, and that is load-bearing rather than an oversight
        # — disabling it at generation restores exactly the over-reliance on the
        # previous frame that the bottleneck exists to break.
        prenet_p = getattr(self.config, "prenet_dropout", 0.0)
        if apply_prenet_dropout and prenet_p > 0.0:
            projected = F.dropout(projected, p=prenet_p, training=True)

        if hasattr(self, 'pos_encoding'):
            projected = self.pos_encoding(projected, offset=position_offset)

        # MegaTransformerEncoderBlock.forward already adds residuals internally,
        # so the loop just chains layers without re-adding the input.
        x = projected
        new_kv_caches = []
        for i, block in enumerate(self.prelude):
            block_cache = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x, new_cache = torch_checkpoint(
                    block, x, None, None, block_cache, position_offset, use_cache,
                    use_reentrant=False,
                )
            else:
                x, new_cache = block(
                    x,
                    kv_cache=block_cache,
                    position_offset=position_offset,
                    use_cache=use_cache,
                )
            if use_cache:
                new_kv_caches.append(new_cache)

        if hasattr(self, 'output_norm'):
            x = self.output_norm(x)

        if use_cache:
            return x, new_kv_caches
        return x
