import math

import torch
import torch.nn as nn

from typing import Optional

from config.audio.feature_extractor import AUDIO_PRELUDE_CONFIGS, AudioVAEPreludeFeatureExtractorConfig
from model.audio.vae.vae import AudioVAEEncoder
from model.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from model.transformer import MegaTransformerBlock
from utils.megatransformer_utils import transformer_weight_init


class AudioVAEPreludeFeatureExtractor(nn.Module):
    """
    Encodes audio into token embeddings for the multimodal world model.

    Supports two modes:
    1. Live encoding: Pass raw mel spectrograms, VAE encoder compresses to latents
    2. Precomputed latents: Pass precomputed VAE latents directly (for efficient training)

    The pipeline after obtaining latents:
    1. Latent channels and compressed mel bins are flattened and projected to d_model
    2. 1D positional encoding is added
    3. Audio-specific prelude transformer processes the sequence

    The prelude allows audio-specific processing before interleaving with other modalities.
    Uses a residual connection around the prelude transformer.
    """

    def __init__(self, config: AudioVAEPreludeFeatureExtractorConfig, vae_encoder: Optional[AudioVAEEncoder] = None):
        super().__init__()

        self.config = config

        self.latent_mels_compressed = config.audio_config.n_mels // config.audio_config.latent_compression_factor[0]

        prelude_config = config.prelude_config
        self.prelude = MegaTransformerBlock(prelude_config)

        self.projection = nn.Linear(
            self.latent_mels_compressed * config.audio_config.latent_channels,
            prelude_config.d_model
        )

        self.vae_encoder = vae_encoder

        max_mel_timesteps = int(
            (config.audio_config.sample_rate * config.audio_config.max_audio_duration)
            / config.audio_config.hop_length
        )
        max_latent_mel_timesteps = max_mel_timesteps // config.audio_config.latent_compression_factor[1]

        # 1D positional encoding for audio timesteps
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=prelude_config.d_model,
            max_len=max_latent_mel_timesteps * 2 + 1,  # Should cover most audio lengths after compression (~60 seconds with default params)
            dropout=0.0
        )

        self._init_weights()

    def _init_weights(self):
        # Only init projection and prelude; VAE encoder has its own init
        # Sinusoidal PE has no learnable params
        self.projection.apply(transformer_weight_init())
        self.prelude.apply(transformer_weight_init())

    @classmethod
    def from_config(cls, config_name: str, vae_encoder: Optional[AudioVAEEncoder]=None, **overrides) -> "AudioVAEPreludeFeatureExtractor":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = AudioVAEPreludeFeatureExtractor.from_config("small", vae_encoder=my_vae_encoder, prelude_config=custom_prelude_config)
        """
        if config_name not in AUDIO_PRELUDE_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(AUDIO_PRELUDE_CONFIGS.keys())}")

        config = AUDIO_PRELUDE_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = AudioVAEPreludeFeatureExtractorConfig(**config_dict)

        return cls(config, vae_encoder=vae_encoder)

    def forward(self, x: torch.Tensor, precomputed_latents: bool = False) -> torch.Tensor:
        """
        Process audio input into token embeddings.

        Args:
            x: Either raw mel spectrograms or precomputed VAE latents.
                - Raw mel specs: (batch_size, mel_bins, timesteps)
                - Precomputed latents: (batch_size, latent_channels, compressed_mel_bins, timesteps)
            precomputed_latents: If True, x is treated as precomputed VAE latents.
                If False, x is passed through the VAE encoder first.

        Returns:
            Token embeddings of shape (batch_size, seq_length, d_model).
        """
        if precomputed_latents:
            latent_representations = x
        else:
            if self.vae_encoder is None:
                raise ValueError("VAE encoder required for raw mel spec input. "
                                 "Either provide vae_encoder at init or use precomputed_latents=True.")
            latent_representations = self.vae_encoder(x)

        # Shape: (batch_size, latent_channels, compressed_mel_bins, timesteps)
        N, C, H, W = latent_representations.size()

        latent_representations = latent_representations.permute(0, 3, 2, 1).contiguous()  # (batch, timesteps, compressed_mel_bins, latent_channels)
        latent_representations = latent_representations.view(N, W, H * C)  # (batch, timesteps, compressed_mel_bins * latent_channels)
        projected_latents = self.projection(latent_representations)  # (batch, timesteps, d_model)

        # Add 1D positional encoding
        projected_latents = self.pos_encoding(projected_latents)

        prelude_hidden, _ = self.prelude(projected_latents)  # (batch, timesteps, d_model)
        prelude_output = projected_latents + prelude_hidden

        return prelude_output
