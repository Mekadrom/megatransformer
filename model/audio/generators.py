import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional

from model.audio.vae import AudioVAEDecoder
from model.world.transformer import MegaTransformerBlock
from utils import configuration
from utils.megatransformer_utils import transformer_weight_init


@dataclass
class AudioCodaAndVAEConfig:
    coda_config: configuration.TransformerBlockConfig
    audio_config: configuration.AudioConfig


class AudioCodaAndVAEWithLoss(nn.Module):
    """
    Audio/voice output head for the multimodal world model.

    Supports two modes:
    1. Latent-only (training): Output latent predictions, compute loss in latent space
    2. Full decode (inference): Decode latents to mel spectrograms via VAE decoder

    The pipeline is:
    1. Coda transformer with residual connection
    2. Linear projection to (latent_mel_bins * latent_channels)
    3. Reshape to VAE latent format (batch, channels, mel_bins, time)
    4. (Optional) VAE decoder produces mel spectrograms

    Works for both general audio and voice - just pass the appropriate VAE decoder.
    Vocoding (mel to waveform) is handled externally.
    """

    def __init__(self, prefix: str, config: AudioCodaAndVAEConfig, vae_decoder: Optional[AudioVAEDecoder] = None):
        super(AudioCodaAndVAEWithLoss, self).__init__()
        
        self.prefix = prefix
        self.config = config
        self.latent_channels = config.audio_config.latent_channels
        self.latent_mel_bins = config.audio_config.n_mels // config.audio_config.latent_compression_factor[0]

        coda_config = config.coda_config
        self.coda = MegaTransformerBlock(coda_config)

        self.latent_projection = nn.Linear(
            coda_config.d_model,
            self.latent_mel_bins * config.audio_config.latent_channels
        )

        self.vae_decoder = vae_decoder

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self._init_weights()

    def _init_weights(self):
        # Only init coda and projection; VAE decoder has its own init
        self.coda.apply(transformer_weight_init())
        self.latent_projection.apply(transformer_weight_init())

    def forward(
        self,
        x: torch.Tensor,
        latent_labels: Optional[torch.Tensor] = None,
        mel_spec_labels: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        decode_to_mel: bool = False,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Process hidden states through coda and optionally decode to mel spectrograms.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            latent_labels: Target latents for loss computation during training.
                Shape: (batch_size, latent_channels, latent_mel_bins, timesteps)
            mel_spec_labels: Target mel spectrograms for loss computation (only used if decode_to_mel=True).
            speaker_embedding: Speaker embedding for voice synthesis.
            lengths: Lengths in latent space timesteps for attention masking.
            decode_to_mel: If True, decode latents to mel spectrograms via VAE decoder.
                If False, only output latent predictions (more efficient for training).

        Returns:
            Dictionary containing:
            - "latent_preds": Predicted latents (batch, channels, mel_bins, time)
            - "reconstructed_mel_specs": Decoded mel specs (only if decode_to_mel=True)
            - "latent_l1_loss", "latent_mse_loss": Latent space losses (if latent_labels provided)
            - "mel_l1_loss", "mel_mse_loss": Mel space losses (if decode_to_mel and mel_spec_labels provided)
        """
        coda_output = x + self.coda(x)  # (batch, seq_length, d_model)

        latent_preds: torch.Tensor = self.latent_projection(coda_output)  # (batch, seq_length, latent_mel_bins * latent_channels)

        N, T, _ = latent_preds.size()
        latent_preds = latent_preds.view(N, T, self.latent_channels, self.latent_mel_bins)
        latent_preds = latent_preds.permute(0, 2, 3, 1)  # (batch, channels, mel_bins, time)

        outputs = {f"{self.prefix}_latent_preds": latent_preds}

        # Latent space loss (for training with precomputed latents)
        if latent_labels is not None:
            outputs[f"{self.prefix}_latent_l1_loss"] = self.l1_loss(latent_preds, latent_labels)
            outputs[f"{self.prefix}_latent_mse_loss"] = self.mse_loss(latent_preds, latent_labels)

        # Optionally decode to mel spectrograms
        if decode_to_mel:
            if self.vae_decoder is None:
                raise ValueError("VAE decoder required for mel decoding. "
                                 "Either provide vae_decoder at init or use decode_to_mel=False.")
            reconstructed_mel_specs = self.vae_decoder(
                latent_preds,
                speaker_embedding=speaker_embedding,
                lengths=lengths,
                return_attention_weights=kwargs.pop("return_attention_weights", False),
                return_film_stats=kwargs.pop("return_film_stats", False),
            )
            outputs[f"{self.prefix}_reconstructed_mel_specs"] = reconstructed_mel_specs

            if mel_spec_labels is not None:
                outputs[f"{self.prefix}_mel_l1_loss"] = self.l1_loss(reconstructed_mel_specs, mel_spec_labels)
                outputs[f"{self.prefix}_mel_mse_loss"] = self.mse_loss(reconstructed_mel_specs, mel_spec_labels)

        return outputs
