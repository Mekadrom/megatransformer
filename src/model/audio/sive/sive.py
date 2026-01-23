from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from config.audio.vae.sive import CONFIGS as SIVE_CONFIGS
from config.audio.vae.sive import SpeakerInvariantVoiceEncoderConfig
from model.audio.sive.conv_subsampling import ConvSubsampling
from model.audio.sive.sive_block import SpeakerInvariantVoiceEncoderBlock


class SpeakerInvariantVoiceEncoder(nn.Module):
    """
    SIVE: Speaker-Invariant Voice Encoder

    Trained with proxy objective:
      - ASR (CTC loss) - forces encoding of linguistic content

    The resulting features are content-focused and speaker-agnostic,
    suitable as targets for a content VAE or direct use in TTS.
    """

    def __init__(self, config: SpeakerInvariantVoiceEncoderConfig):
        super().__init__()

        self.config = config

        # Convolutional subsampling frontend with optional Dropout1d
        self.conv_subsample = ConvSubsampling(
            in_channels=config.n_mels,
            out_channels=config.encoder_dim,
            kernel_sizes=config.conv_kernel_sizes,
            strides=config.conv_strides,
            dropout=config.conv_dropout if config.conv_dropout > 0 else config.dropout,
        )

        # Transformer encoder blocks with architectural options
        self.encoder_blocks = nn.ModuleList([
            SpeakerInvariantVoiceEncoderBlock(
                d_model=config.encoder_dim,
                n_heads=config.num_heads,
                d_ff=config.ff_dim,
                dropout=config.dropout,
                head_drop_prob=config.attention_head_drop,
                conformer_kernel_size=config.conformer_kernel_size,
                activation=config.activation,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.encoder_dim)

        # Feature dropout (applied before heads)
        self.feature_dropout = nn.Dropout(config.feature_dropout) if config.feature_dropout > 0 else nn.Identity()

        self.asr_head = nn.Sequential(
            nn.Linear(config.encoder_dim, config.encoder_dim),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.encoder_dim, config.vocab_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "SpeakerInvariantVoiceEncoder":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = SpeakerInvariantVoiceEncoder.from_config("small", num_speakers=500)
        """
        if config_name not in SIVE_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(SIVE_CONFIGS.keys())}")

        config = SIVE_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = SpeakerInvariantVoiceEncoderConfig(**config_dict)

        return cls(config)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input mel spectrogram length."""
        return self.conv_subsample.get_output_length(input_length)

    def forward(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
    ) -> dict:
        """
        Forward pass through SIVE.

        Args:
            mel_spec: [B, n_mels, T] mel spectrogram
            lengths: [B] original lengths before padding (optional)
            return_all_hiddens: If True, return hidden states from all layers

        Returns:
            dict with keys:
                - features: [B, T', encoder_dim] normalized content features
                - features_unnorm: [B, T', encoder_dim] pre-LayerNorm features
                - asr_logits: [B, T', vocab_size] for CTC loss
                - feature_lengths: [B] output sequence lengths
                - variance_loss: scalar loss for variance regularization (if enabled)
                - temporal_smoothness: scalar metric (if variance_reg enabled and training)
                - all_hiddens: list of [B, T', D] if return_all_hiddens=True
        """
        # Convolutional frontend
        x: torch.Tensor = self.conv_subsample(mel_spec)  # [B, encoder_dim, T']
        x = x.permute(0, 2, 1)  # [B, T', encoder_dim]

        # Calculate output lengths
        feature_lengths = None
        padding_mask = None
        if lengths is not None:
            feature_lengths = torch.tensor(
                [self.get_output_length(l.item()) for l in lengths],
                device=mel_spec.device,
                dtype=torch.long,
            )
            # Create padding mask [B, T'] - True for padded positions
            max_len = x.size(1)
            padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= feature_lengths.unsqueeze(1)

        # Transformer encoder
        all_hiddens = [x] if return_all_hiddens else None
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=padding_mask)

            if return_all_hiddens:
                all_hiddens.append(x)

        # Store pre-LayerNorm features
        features_unnorm = x

        # Final normalization
        features = self.final_norm(x)

        # Compute variance regularization loss (for VAE-friendly features)
        variance_loss = torch.tensor(0.0, device=mel_spec.device)
        temporal_smoothness = None
        if self.config.use_variance_reg:
            # Use normalized features for variance computation
            feat_for_var = features

            # Temporal variance: want features to change over time (not constant)
            if feat_for_var.size(1) > 1:
                temporal_diff = feat_for_var[:, 1:, :] - feat_for_var[:, :-1, :]
                temporal_var = temporal_diff.var(dim=1).mean()
                temporal_loss = F.relu(self.config.temporal_var_min - temporal_var)

                # Temporal smoothness (for logging) - cosine similarity between adjacent frames
                with torch.no_grad():
                    feat_norm = F.normalize(feat_for_var, dim=-1)
                    cos_sim = (feat_norm[:, 1:, :] * feat_norm[:, :-1, :]).sum(dim=-1)
                    temporal_smoothness = cos_sim.mean()
            else:
                temporal_loss = torch.tensor(0.0, device=mel_spec.device)

            # Per-dimension variance: want each dimension to be used (not dead)
            dim_var = feat_for_var.var(dim=(0, 1))
            dim_loss = F.relu(self.config.dim_var_min - dim_var).mean()

            # Temporal smoothness penalty: penalize if features are too smooth
            smoothness_loss = torch.tensor(0.0, device=mel_spec.device)
            if temporal_smoothness is not None:
                smoothness_loss = F.relu(temporal_smoothness - self.config.temporal_smoothness_max)

            variance_loss = (
                self.config.temporal_var_weight * temporal_loss +
                self.config.dim_var_weight * dim_loss +
                self.config.temporal_smoothness_weight * smoothness_loss
            )

        # Apply feature dropout before heads
        features_for_heads = self.feature_dropout(features)

        # CTC upsampling: upsample features before ASR head to relax CTC length constraint
        # This keeps transformer efficient (operates on T/4) while giving CTC more frames
        ctc_lengths = feature_lengths
        if self.config.ctc_upsample_factor > 1:
            # Linear interpolation upsampling: [B, T, D] -> [B, T*factor, D]
            features_for_asr = features_for_heads.transpose(1, 2)  # [B, D, T]
            features_for_asr = F.interpolate(
                features_for_asr,
                scale_factor=float(self.config.ctc_upsample_factor),
                mode='linear',
                align_corners=False,
            )
            features_for_asr = features_for_asr.transpose(1, 2)  # [B, T*factor, D]

            # Update CTC lengths to reflect upsampling
            if feature_lengths is not None:
                ctc_lengths = feature_lengths * self.config.ctc_upsample_factor
        else:
            features_for_asr = features_for_heads

        # ASR head (operates on potentially upsampled features)
        asr_logits = self.asr_head(features_for_asr)

        result = {
            "features": features,  # Normalized features (preferred for VAE)
            "features_unnorm": features_unnorm,  # Pre-LayerNorm (for comparison)
            "asr_logits": asr_logits,
            "feature_lengths": feature_lengths,  # Original feature lengths (for feature extraction)
            "ctc_lengths": ctc_lengths,  # CTC lengths (potentially upsampled, for CTC loss)
            "variance_loss": variance_loss,
            "temporal_smoothness": temporal_smoothness if self.config.use_variance_reg and self.training else None,
        }

        if return_all_hiddens:
            result["all_hiddens"] = all_hiddens

        return result

    def extract_features(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        layer: int = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features without computing ASR head output.
        Useful for inference after training.

        Args:
            mel_spec: [B, n_mels, T]
            lengths: [B] optional lengths
            layer: Which layer to extract from (-1 = final)

        Returns:
            features: [B, T', encoder_dim]
            feature_lengths: [B] or None
        """
        with torch.no_grad():
            result = self.forward(
                mel_spec,
                lengths=lengths,
                return_all_hiddens=(layer != -1),
            )

        if layer == -1:
            features = result["features"]
        else:
            features = result["all_hiddens"][layer]

        return features, result["feature_lengths"]

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
