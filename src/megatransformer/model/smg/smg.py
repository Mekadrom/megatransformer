import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union

from torch.utils.checkpoint import checkpoint

from megatransformer.config.smg.smg import SMG_CONFIGS, SMG_DECODER_2D_CONFIGS, SMG_DECODER_1D_CONFIGS, F0_CONDITIONING_EMBEDDING_CONFIGS, F0_PREDICTOR_CONFIGS, SMGConfig, F0ConditioningEmbeddingConfig, F0PredictorConfig
from megatransformer.config.smg.smg import SMGDecoder2DConfig, SMGDecoder1DConfig
from megatransformer.model import activations
from megatransformer.model.activations import get_activation_type
from megatransformer.model.smg.residual_block import ResidualBlock1d, ResidualBlock2d


class SMGDecoder2D(nn.Module):
    """
    SMG (SIVE-Mel Generator) 2D decoder that transitions from 1D (SIVE features latents) to 2D convolutions (mel specs).

    Takes 1D latent [B, latent_dim, T'] and produces 2D mel spectrogram [B, 1, n_mels, T].

    Architecture:
    1. 1D processing: latent → Conv1d stages with optional upsampling
    2. Reshape: [B, C, T''] → [B, C', initial_freq_bins, T'']
    3. 2D processing: Conv2d stages that upsample both frequency and time
    4. Output: [B, 1, n_mels, T]

    The 1D→2D transition allows the model to:
    - Process temporal dynamics with 1D convs (efficient, appropriate for latent)
    - Generate mel spectrograms with 2D convs (captures frequency-local patterns like harmonics)
    """
    def __init__(self, config: SMGDecoder2DConfig):
        super().__init__()

        self.config = config
        self.gradient_checkpointing = False

        # Effective speaker dim for FiLM
        self.film_speaker_dim = config.speaker_embedding_proj_dim if config.speaker_embedding_proj_dim > 0 else config.speaker_embedding_dim

        # Input dim is the SIVE encoder feature width (F0 is injected after 1D→2D transition)
        decoder_input_dim = config.sive_encoder_dim

        self.scale_factors_2d = config.scale_factors_2d

        # Compute channels after 1D→2D reshape
        channels_2d_initial = config.conv1d_channels // config.initial_freq_bins
        self.channels_2d_initial = channels_2d_initial
        assert config.conv1d_channels % config.initial_freq_bins == 0, \
            f"conv1d_channels ({config.conv1d_channels}) must be divisible by initial_freq_bins ({config.initial_freq_bins})"

        # F0 conditioning projection: inject after 1D→2D transition
        # Projects F0 embedding to 2D spatial dimensions for additive conditioning
        # Project [B, f0_dim, T'] → [B, channels_2d * freq_bins, T'] → reshape to [B, C, H, T']
        self.f0_to_2d_projection = nn.Sequential(
            nn.Conv1d(config.f0_embedding_dim, channels_2d_initial * config.initial_freq_bins, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels_2d_initial * config.initial_freq_bins, channels_2d_initial * config.initial_freq_bins, kernel_size=3, padding=1),
        )

        # Speaker embedding projection
        self.speaker_embedding_projection = None
        if config.speaker_embedding_dim > 0 and config.speaker_embedding_proj_dim > 0 and config.speaker_embedding_proj_dim != config.speaker_embedding_dim:
            self.speaker_embedding_projection = nn.Sequential(
                nn.Linear(config.speaker_embedding_dim, config.speaker_embedding_proj_dim),
                nn.SiLU(),
            )

        # === 1D Processing Stage ===
        # Activation helper
        if config.activation == "snake":
            self.get_activation_1d = lambda c: activations.Snake(c)
            self.get_activation_2d = lambda c: activations.Snake2d(c)
        else:
            activation_type = get_activation_type(config.activation)
            if activation_type in [activations.SwiGLU, activations.Snake]:
                self.get_activation_1d = lambda c: activation_type(c)
                self.get_activation_2d = lambda c: activation_type(c)
            else:
                self.get_activation_1d = lambda _: activation_type()
                self.get_activation_2d = lambda _: activation_type()

        # Optional InstanceNorm on the RAW input features (strips per-channel
        # utterance moments = speaker envelope before any learned mixing). Off by default.
        self.input_instance_norm = (
            nn.InstanceNorm1d(decoder_input_dim, affine=False)
            if getattr(config, "input_instance_norm", False) else None
        )

        # Initial 1D projection
        self.conv1d_initial = nn.Conv1d(decoder_input_dim, config.conv1d_channels, kernel_size=3, padding=1)
        self.conv1d_initial_norm = nn.GroupNorm(max(1, config.conv1d_channels // 4), config.conv1d_channels)
        self.conv1d_initial_act = self.get_activation_1d(config.conv1d_channels)

        # Optional 1D upsampling
        self.conv1d_upsample = None
        if config.conv1d_upsample_factor > 1:
            self.conv1d_upsample = nn.Sequential(
                nn.Upsample(scale_factor=config.conv1d_upsample_factor, mode='nearest'),
                nn.Conv1d(config.conv1d_channels, config.conv1d_channels, kernel_size=config.conv1d_kernel_size, padding=config.conv1d_kernel_size // 2),
                nn.GroupNorm(max(1, config.conv1d_channels // 4), config.conv1d_channels),
                self.get_activation_1d(config.conv1d_channels),
            )

        # 1D residual blocks
        self.conv1d_residual_blocks = nn.ModuleList([
            ResidualBlock1d(config.conv1d_channels, config.conv1d_channels, kernel_size=config.conv1d_kernel_size, activation_fn=config.activation)
            for _ in range(config.conv1d_n_residual_blocks)
        ])

        # Early FiLM (in 1D stage)
        if config.speaker_embedding_dim > 0:
            self.early_film_projection = nn.Sequential(
                nn.Linear(self.film_speaker_dim, self.film_speaker_dim),
                nn.SiLU(),
                nn.Linear(self.film_speaker_dim, config.conv1d_channels * 2),
            )

        # === 2D Processing Stages ===
        # FiLM projections for 2D stages
        if config.speaker_embedding_dim > 0:
            self.speaker_projections_2d = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.film_speaker_dim, self.film_speaker_dim),
                    nn.SiLU(),
                    nn.Linear(self.film_speaker_dim, out_c * 2),
                )
                for out_c in config.intermediate_channels_2d
            ])

        # Build 2D stages
        self.stages_2d = nn.ModuleList()
        all_channels_2d = [channels_2d_initial] + config.intermediate_channels_2d

        for in_c, out_c, kernel_size, scale_factor in zip(
            all_channels_2d[:-1], config.intermediate_channels_2d, config.kernel_sizes_2d, config.scale_factors_2d
        ):
            stage = nn.ModuleList()

            # Upsample (freq, time)
            stage.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))

            # Conv2d after upsample
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            stage.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding))
            stage.append(nn.GroupNorm(max(1, out_c // 4), out_c))
            stage.append(self.get_activation_2d(out_c))

            # Residual blocks
            for _ in range(config.n_residual_blocks_2d):
                stage.append(ResidualBlock2d(out_c, out_c, kernel_size=kernel_size, activation_fn=config.activation))

            if config.dropout > 0:
                stage.append(nn.Dropout2d(config.dropout))

            self.stages_2d.append(stage)

        # Final output conv
        self.final_conv = nn.Conv2d(config.intermediate_channels_2d[-1], 1, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Init FiLM projections
        if self.config.speaker_embedding_dim > 0:
            if hasattr(self, 'early_film_projection'):
                self._init_film_projection(self.early_film_projection)
            for proj in self.speaker_projections_2d:
                self._init_film_projection(proj)

    def _init_film_projection(self, proj):
        """Initialize a FiLM projection module."""
        if isinstance(proj, nn.Sequential):
            linear_layers = [m for m in proj if isinstance(m, nn.Linear)]
            for i, linear in enumerate(linear_layers):
                if i < len(linear_layers) - 1:
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)
                else:
                    nn.init.normal_(linear.weight, mean=0.0, std=0.02)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)

    @classmethod
    def from_config(cls, config: Union[str, SMGDecoder2DConfig], **overrides) -> "SMGDecoder2D":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = SMGDecoder2D.from_config("small", latent_dim=32)
        """
        if isinstance(config, str):
            if config not in SMG_DECODER_2D_CONFIGS:
                raise ValueError(f"Unknown config: {config}. Available: {list(SMG_DECODER_2D_CONFIGS.keys())}")
            config = SMG_DECODER_2D_CONFIGS[config]

        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = SMGDecoder2DConfig(**config_dict)

        return cls(config)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def get_output_length(self, input_length: int) -> int:
        """Compute output temporal length given latent length."""
        length = input_length
        if self.config.conv1d_upsample_factor > 1:
            length = length * self.config.conv1d_upsample_factor
        for scale_factor in self.config.scale_factors_2d:
            length = length * scale_factor[1]  # Time dimension
        return length

    def _run_stage_2d(self, stage: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for layer in stage:
            x = layer(x)
        return x

    def forward(
        self,
        z: torch.Tensor,
        speaker_embedding: torch.Tensor,
        f0_embedding: torch.Tensor,
        return_film_stats: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        """
        Args:
            z: [B, latent_dim, T'] latent tensor (F0 is NOT concatenated here)
            speaker_embedding: [B, speaker_dim] or [B, 1, speaker_dim] speaker embedding for FiLM
            f0_embedding: [B, f0_conditioning_dim, T'] F0 conditioning (injected after 1D→2D transition)

        Returns:
            recon: [B, 1, n_mels, T] reconstructed mel spectrogram
        """
        film_stats: Optional[dict[str, float]] = {} if return_film_stats else None

        # Process speaker embedding
        if speaker_embedding is not None:
            if speaker_embedding.dim() == 3:
                speaker_embedding = speaker_embedding.squeeze(1)
            if self.config.normalize_speaker_embedding:
                speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)
            if self.speaker_embedding_projection is not None:
                speaker_embedding = self.speaker_embedding_projection(speaker_embedding)

        # === 1D Processing ===
        # Optional input InstanceNorm — strip the speaker envelope from the raw
        # SIVE features before any learned mixing (see __init__).
        if self.input_instance_norm is not None:
            z = self.input_instance_norm(z)
        x: torch.Tensor = self.conv1d_initial(z)
        x = self.conv1d_initial_norm(x)
        x = self.conv1d_initial_act(x)

        # Early FiLM (1D)
        if speaker_embedding is not None and self.config.speaker_embedding_dim > 0:
            film_params = self.early_film_projection(speaker_embedding)
            channels = x.shape[1]
            scale = film_params[:, :channels].unsqueeze(-1)
            shift = film_params[:, channels:].unsqueeze(-1)

            if return_film_stats:
                film_stats["early_scale_raw_mean"] = scale.mean().item()
                film_stats["early_shift_raw_mean"] = shift.mean().item()

            if self.config.film_scale_bound > 0:
                scale = self.config.film_scale_bound * torch.tanh(scale)
            if self.config.film_shift_bound > 0:
                shift = self.config.film_shift_bound * torch.tanh(shift)

            x = x * (1 + scale) + shift

        # Optional 1D upsampling
        if self.conv1d_upsample is not None:
            x = self.conv1d_upsample(x)

        # 1D residual blocks
        for block in self.conv1d_residual_blocks:
            x = block(x)

        # === 1D → 2D Transition ===
        # x: [B, conv1d_channels, T''] → [B, channels_2d, initial_freq_bins, T'']
        B, C, T = x.shape
        channels_2d = C // self.config.initial_freq_bins
        x = x.view(B, channels_2d, self.config.initial_freq_bins, T)

        # Inject F0 conditioning after reshape to 2D (additive)
        # F0 can now influence frequency-specific features
        # Match temporal dimension if needed (after any 1D upsampling)
        if f0_embedding.shape[-1] != T:
            f0_embedding = F.interpolate(f0_embedding, size=T, mode='linear', align_corners=False)
        # Project F0 to 2D: [B, f0_dim, T] → [B, C*H, T] → [B, C, H, T]
        f0_2d: torch.Tensor = self.f0_to_2d_projection(f0_embedding)  # [B, C*H, T]
        f0_2d = f0_2d.view(B, self.channels_2d_initial, self.config.initial_freq_bins, T)
        x = x + f0_2d

        # === 2D Processing ===
        for i, stage in enumerate(self.stages_2d):
            if self.gradient_checkpointing and self.training and i >= 1:
                x = checkpoint(self._run_stage_2d, stage, x, use_reentrant=False)
            else:
                x = self._run_stage_2d(stage, x)

            # FiLM conditioning after each 2D stage
            if speaker_embedding is not None and self.config.speaker_embedding_dim > 0:
                film_params: torch.Tensor = self.speaker_projections_2d[i](speaker_embedding)
                out_c = x.shape[1]
                scale = film_params[:, :out_c].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                shift = film_params[:, out_c:].unsqueeze(-1).unsqueeze(-1)

                if return_film_stats:
                    film_stats[f"stage2d_{i}_scale_raw_mean"] = scale.mean().item()
                    film_stats[f"stage2d_{i}_shift_raw_mean"] = shift.mean().item()

                if self.config.film_scale_bound > 0:
                    scale = self.config.film_scale_bound * torch.tanh(scale)
                if self.config.film_shift_bound > 0:
                    shift = self.config.film_shift_bound * torch.tanh(shift)

                x = x * (1 + scale) + shift

        # Final output
        x = self.final_conv(x)  # [B, 1, n_mels, T]

        if return_film_stats:
            return x, film_stats

        return x


class SMGDecoder1D(nn.Module):
    """
    Conv1D-only SMG (SIVE-Mel Generator) decoder. Treats frequency as channels throughout.

    Avoids ring/circle artifacts caused by Conv2D's spatial isotropy assumption
    on mel spectrograms where frequency and time axes have different semantics.

    Takes 1D latent [B, latent_dim, T'] and produces mel spectrogram [B, output_dim, T].

    Architecture:
    1. Initial projection: latent → initial_channels
    2. F0 injection (additive)
    3. Early FiLM (speaker conditioning)
    4. Pre-upsample residual blocks
    5. Upsample stages: Upsample + Conv1d + ResidualBlock1d + FiLM
    6. Final conv: channels → output_dim (80 mel bins)
    """
    def __init__(self, config: SMGDecoder1DConfig):
        super().__init__()

        self.config = config
        self.gradient_checkpointing = False

        # Effective speaker dim for FiLM
        self.film_speaker_dim = config.speaker_embedding_proj_dim if config.speaker_embedding_proj_dim > 0 else config.speaker_embedding_dim

        # Activation helper
        if config.activation == "snake":
            self.get_activation = lambda c: activations.Snake(c)
        else:
            activation_type = get_activation_type(config.activation)
            if activation_type in [activations.SwiGLU, activations.Snake]:
                self.get_activation = lambda c: activation_type(c)
            else:
                self.get_activation = lambda _: activation_type()

        # Speaker embedding projection
        self.speaker_embedding_projection = None
        if config.speaker_embedding_dim > 0 and config.speaker_embedding_proj_dim > 0 and config.speaker_embedding_proj_dim != config.speaker_embedding_dim:
            self.speaker_embedding_projection = nn.Sequential(
                nn.Linear(config.speaker_embedding_dim, config.speaker_embedding_proj_dim),
                nn.SiLU(),
            )

        # Optional InstanceNorm on the RAW input features (before the initial conv):
        # strips per-channel utterance moments (speaker's global spectral envelope)
        # so the SMG must source that detail from the embedding. affine=False =
        # genuinely lossy (no learnable restore). Off by default.
        self.input_instance_norm = (
            nn.InstanceNorm1d(config.sive_encoder_dim, affine=False)
            if getattr(config, "input_instance_norm", False) else None
        )

        # Initial projection from latent to working channels
        self.initial_conv = nn.Conv1d(config.sive_encoder_dim, config.initial_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(max(1, config.initial_channels // 4), config.initial_channels)
        self.initial_act = self.get_activation(config.initial_channels)

        # F0 conditioning injection (additive, projected to initial_channels)
        self.f0_projection = nn.Sequential(
            nn.Conv1d(config.f0_embedding_dim, config.initial_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(config.initial_channels, config.initial_channels, kernel_size=3, padding=1),
        )

        # Early FiLM (speaker conditioning on initial channels)
        if config.speaker_embedding_dim > 0:
            self.early_film_projection = nn.Sequential(
                nn.Linear(self.film_speaker_dim, self.film_speaker_dim),
                nn.SiLU(),
                nn.Linear(self.film_speaker_dim, config.initial_channels * 2),
            )

        # Pre-upsample residual blocks
        self.pre_upsample_blocks = nn.ModuleList([
            ResidualBlock1d(
                config.initial_channels, config.initial_channels,
                kernel_size=config.pre_upsample_kernel_size,
                activation_fn=config.activation,
            )
            for _ in range(config.pre_upsample_residual_blocks)
        ])

        # Upsample stages
        all_channels = [config.initial_channels] + config.stage_channels
        self.upsample_stages = nn.ModuleList()

        for in_c, out_c, kernel_size, upsample_factor in zip(
            all_channels[:-1], config.stage_channels, config.stage_kernel_sizes, config.time_upsample_factors
        ):
            stage = nn.ModuleList()

            # Upsample (skip if factor=1)
            if upsample_factor > 1:
                stage.append(nn.Upsample(scale_factor=upsample_factor, mode='nearest'))
            else:
                stage.append(nn.Identity())

            # Conv after upsample
            padding = kernel_size // 2
            stage.append(nn.Conv1d(in_c, out_c, kernel_size, padding=padding))
            stage.append(nn.GroupNorm(max(1, out_c // 4), out_c))
            stage.append(self.get_activation(out_c))

            # Residual blocks
            for _ in range(config.n_residual_blocks_per_stage):
                stage.append(ResidualBlock1d(out_c, out_c, kernel_size=kernel_size, activation_fn=config.activation))

            if config.dropout > 0:
                stage.append(nn.Dropout(config.dropout))

            self.upsample_stages.append(stage)

        # FiLM projections for upsample stages
        if config.speaker_embedding_dim > 0:
            self.stage_film_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.film_speaker_dim, self.film_speaker_dim),
                    nn.SiLU(),
                    nn.Linear(self.film_speaker_dim, out_c * 2),
                )
                for out_c in config.stage_channels
            ])

        # Final output conv — named final_conv for adaptive weight compatibility
        self.final_conv = nn.Conv1d(config.stage_channels[-1], config.output_dim, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Init FiLM projections
        if self.config.speaker_embedding_dim > 0:
            if hasattr(self, 'early_film_projection'):
                self._init_film_projection(self.early_film_projection)
            for proj in self.stage_film_projections:
                self._init_film_projection(proj)

    def _init_film_projection(self, proj):
        if isinstance(proj, nn.Sequential):
            linear_layers = [m for m in proj if isinstance(m, nn.Linear)]
            for i, linear in enumerate(linear_layers):
                if i < len(linear_layers) - 1:
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)
                else:
                    nn.init.normal_(linear.weight, mean=0.0, std=0.02)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)

    @classmethod
    def from_config(cls, config: Union[str, SMGDecoder1DConfig], **overrides) -> "SMGDecoder1D":
        if isinstance(config, str):
            if config not in SMG_DECODER_1D_CONFIGS:
                raise ValueError(f"Unknown config: {config}. Available: {list(SMG_DECODER_1D_CONFIGS.keys())}")
            config = SMG_DECODER_1D_CONFIGS[config]

        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = SMGDecoder1DConfig(**config_dict)

        return cls(config)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def get_output_length(self, input_length: int) -> int:
        length = input_length
        for factor in self.config.time_upsample_factors:
            length = length * factor
        return length

    def _apply_film(self, x: torch.Tensor, film_params: torch.Tensor, return_film_stats: bool, stats_dict: Optional[dict], prefix: str) -> torch.Tensor:
        channels = x.shape[1]
        scale = film_params[:, :channels].unsqueeze(-1)  # [B, C, 1]
        shift = film_params[:, channels:].unsqueeze(-1)

        if return_film_stats and stats_dict is not None:
            stats_dict[f"{prefix}_scale_raw_mean"] = scale.mean().item()
            stats_dict[f"{prefix}_shift_raw_mean"] = shift.mean().item()

        if self.config.film_scale_bound > 0:
            scale = self.config.film_scale_bound * torch.tanh(scale)
        if self.config.film_shift_bound > 0:
            shift = self.config.film_shift_bound * torch.tanh(shift)

        return x * (1 + scale) + shift

    def _run_stage(self, stage: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for layer in stage:
            x = layer(x)
        return x

    def forward(
        self,
        z: torch.Tensor,
        speaker_embedding: torch.Tensor,
        f0_embedding: torch.Tensor,
        return_film_stats: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        """
        Args:
            z: [B, latent_dim, T'] latent tensor
            speaker_embedding: [B, speaker_dim] or [B, 1, speaker_dim]
            f0_embedding: [B, f0_conditioning_dim, T'] F0 conditioning

        Returns:
            recon: [B, output_dim, T] reconstructed mel spectrogram (3D, no channel squeeze needed)
        """
        film_stats: Optional[dict[str, float]] = {} if return_film_stats else None

        # Process speaker embedding
        if speaker_embedding is not None:
            if speaker_embedding.dim() == 3:
                speaker_embedding = speaker_embedding.squeeze(1)
            if self.config.normalize_speaker_embedding:
                speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)
            if self.speaker_embedding_projection is not None:
                speaker_embedding = self.speaker_embedding_projection(speaker_embedding)

        # Optional input InstanceNorm — strip the speaker envelope from the raw
        # SIVE features before any learned mixing (see __init__).
        if self.input_instance_norm is not None:
            z = self.input_instance_norm(z)

        # Initial projection
        x: torch.Tensor = self.initial_conv(z)
        x = self.initial_norm(x)
        x = self.initial_act(x)

        # F0 injection (additive)
        if f0_embedding.shape[-1] != x.shape[-1]:
            f0_embedding = F.interpolate(f0_embedding, size=x.shape[-1], mode='linear', align_corners=False)
        f0_cond: torch.Tensor = self.f0_projection(f0_embedding)
        x = x + f0_cond

        # Early FiLM
        if speaker_embedding is not None and self.config.speaker_embedding_dim > 0:
            film_params = self.early_film_projection(speaker_embedding)
            x = self._apply_film(x, film_params, return_film_stats, film_stats, "early")

        # Pre-upsample residual blocks
        for block in self.pre_upsample_blocks:
            x = block(x)

        # Upsample stages with FiLM
        for i, stage in enumerate(self.upsample_stages):
            if self.gradient_checkpointing and self.training and i >= 1:
                x = checkpoint(self._run_stage, stage, x, use_reentrant=False)
            else:
                x = self._run_stage(stage, x)

            # FiLM conditioning after each stage
            if speaker_embedding is not None and self.config.speaker_embedding_dim > 0:
                film_params = self.stage_film_projections[i](speaker_embedding)
                x = self._apply_film(x, film_params, return_film_stats, film_stats, f"stage_{i}")

        # Final output: [B, output_dim, T]
        x = self.final_conv(x)

        if return_film_stats:
            return x, film_stats

        return x


class F0Predictor(nn.Module):
    """
    Predicts F0 (fundamental frequency) contour from speaker embedding + SIVE features.

    Speaker embedding provides F0 range/characteristics (who is speaking).
    SIVE features provide prosodic timing cues (where stress/emphasis occurs).

    Output is per-frame: log F0 values and voiced/unvoiced probability.

    Those two outputs are determined by DIFFERENT inputs, so when vuv_encoder_dim is set
    they get different trunks:

      pitch   needs contour + speaker. Content cannot supply it: an SMG trained on
              prosody-free units plateaued at f0_loss ~0.09 predicting F0 from full
              256-dim ContentVec, while the contour path reaches ~0.03. Worse, letting
              the F0 head see content is an active hazard -- it is the shortcut that
              produced the collapsed F0 path in the first place, so the F0 trunk here
              reads ONLY sive_features (the contour) and never content_features. This
              is why the branches are separate rather than one trunk over a concat: a
              shared trunk would leak content into the F0 head via its receptive field.

      voicing needs content. Voicing is phonemic (/s/ vs /z/), so ContentVec must carry
              it. The contour cannot: measured on 869k val frames, H(vuv)=0.691 and
              H(vuv|contour)=0.677 -- the contour explains 2% of voicing entropy, while
              a bare unit id with no temporal context explains 23% (H(vuv|unit)=0.530).
              A contour-fed voicing head is reading a channel that structurally does not
              contain its answer.

    Voicing gates harmonic content in F0ConditioningEmbedding, so its errors are audible
    as wrong harmonic energy, not merely a bad number.
    """
    def __init__(self, config: F0PredictorConfig):
        super().__init__()

        self.config = config

        # Project speaker embedding to F0-relevant representation
        self.speaker_proj = nn.Linear(config.speaker_embedding_dim, config.hidden_dim)

        # Project SIVE features (prosodic timing info)
        self.sive_proj = nn.Conv1d(config.encoder_dim, config.hidden_dim, kernel_size=1)

        # Combine and predict F0 contour with residual convolutions
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(config.hidden_dim, config.hidden_dim, config.kernel_size, padding=config.kernel_size // 2),
                nn.GroupNorm(8, config.hidden_dim),
                nn.SiLU(),
            )
            for _ in range(config.n_layers)
        ])

        self.vuv_encoder_dim = getattr(config, 'vuv_encoder_dim', None)
        if self.vuv_encoder_dim is not None:
            # Voicing gets its own trunk over content. Speaker is included because the
            # reference point for this head -- the features-fed predictor that reached
            # vuv_loss 0.42 -- had both, and speaker habits (creak, final devoicing) do
            # move the soft periodicity target.
            self.vuv_speaker_proj = nn.Linear(config.speaker_embedding_dim, config.hidden_dim)
            self.vuv_proj = nn.Conv1d(self.vuv_encoder_dim, config.hidden_dim, kernel_size=1)
            self.vuv_conv_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(config.hidden_dim, config.hidden_dim, config.kernel_size, padding=config.kernel_size // 2),
                    nn.GroupNorm(8, config.hidden_dim),
                    nn.SiLU(),
                )
                for _ in range(config.n_layers)
            ])
            self.vuv_out = nn.Conv1d(config.hidden_dim, 1, kernel_size=1)
            # Only log_f0 comes off the shared trunk now.
            self.output_proj = nn.Conv1d(config.hidden_dim, 1, kernel_size=1)
        else:
            # Output: log_f0 (continuous) + voiced_logit
            self.output_proj = nn.Conv1d(config.hidden_dim, 2, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # Initialize output layer to produce reasonable F0 values
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            # Initialize log_f0 bias to ~5.0 (≈150 Hz), voiced bias to 0
            self.output_proj.bias.data[0] = 5.0  # log(150) ≈ 5.0
            if self.output_proj.out_channels > 1:
                self.output_proj.bias.data[1] = 0.0
        if self.vuv_encoder_dim is not None:
            nn.init.zeros_(self.vuv_out.weight)
            if self.vuv_out.bias is not None:
                self.vuv_out.bias.data[0] = 0.0

    @classmethod
    def from_config(cls, config: str, **overrides) -> "F0Predictor":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = F0Predictor.from_config("small", encoder_dim=256)
        """
        if isinstance(config, str):
            if config not in F0_PREDICTOR_CONFIGS:
                raise ValueError(f"Unknown config: {config}. Available: {list(F0_PREDICTOR_CONFIGS.keys())}")
            config = F0_PREDICTOR_CONFIGS[config]

        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = F0PredictorConfig(**config_dict)

        return cls(config)

    def forward(
        self,
        speaker_embedding: torch.Tensor,
        sive_features: torch.Tensor,
        content_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            speaker_embedding: [B, speaker_dim] or [B, 1, speaker_dim]
            sive_features: [B, sive_dim, T'] - what the F0 head reads (contour, or
                content in the legacy "features" path). NEVER content when a vuv branch
                is configured; see the class docstring.
            content_features: [B, vuv_encoder_dim, T] - content for the voicing branch.
                Required iff vuv_encoder_dim is set. Deliberately kept out of the F0
                trunk.

        Returns:
            log_f0: [B, T'] - log fundamental frequency (in log Hz)
            voiced_prob: [B, T'] - probability frame is voiced (0-1)
        """
        if speaker_embedding.dim() == 3:
            speaker_embedding = speaker_embedding.squeeze(1)

        T = sive_features.size(-1)
        # Speaker embedding -> F0 characteristics, broadcast across time
        spk: torch.Tensor = self.speaker_proj(speaker_embedding)  # [B, hidden]
        spk = spk.unsqueeze(-1).expand(-1, -1, T)   # [B, hidden, T']

        # sive -> prosodic timing patterns
        content = self.sive_proj(sive_features)  # [B, hidden, T']

        # Combine: speaker provides "base F0", content provides "modulation"
        x = spk + content

        # Refine with residual convolutions
        for conv in self.conv_layers:
            x = x + conv(x)

        # Predict F0 and voicing
        out = self.output_proj(x)  # [B, 1 or 2, T']
        log_f0 = out[:, 0, :]      # [B, T']

        if self.vuv_encoder_dim is None:
            voiced_prob = torch.sigmoid(out[:, 1, :])  # [B, T']
            return log_f0, voiced_prob

        if content_features is None:
            # Same reasoning as SMG._f0_predictor_input's contour raise: silently
            # degrading here would leave the voicing head reading nothing, and its
            # output gates harmonics -- the failure would sound like a modelling
            # problem rather than a missing argument.
            raise ValueError(
                "vuv_encoder_dim is set (voicing reads content on its own branch) but "
                "no content_features were passed. Pass content_features=; the F0 head "
                "still reads only sive_features."
            )
        if content_features.size(-1) != T:
            content_features = F.interpolate(content_features, size=T, mode='linear', align_corners=False)
        v = self.vuv_speaker_proj(speaker_embedding).unsqueeze(-1).expand(-1, -1, T) \
            + self.vuv_proj(content_features)
        for conv in self.vuv_conv_layers:
            v = v + conv(v)
        voiced_prob = torch.sigmoid(self.vuv_out(v)[:, 0, :])  # [B, T']

        return log_f0, voiced_prob


class F0ConditioningEmbedding(nn.Module):
    """
    Converts F0 predictions into embeddings for decoder conditioning.

    Creates harmonic sinusoidal embeddings that give the decoder explicit information
    about where harmonic energy should appear in the mel spectrogram:
    - sin(2π * h * f0 * t) and cos(2π * h * f0 * t) for each harmonic h
    - Gated by voicing probability (unvoiced frames get zero harmonic content)
    - Projected to embedding dimension

    The sinusoidal representation helps the decoder place harmonics at the correct
    frequency locations, which is crucial for natural-sounding speech synthesis.
    """
    def __init__(self, config: F0ConditioningEmbeddingConfig):
        super().__init__()

        self.config = config

        # Time step in seconds for each frame
        self.frame_duration = config.hop_length / config.sample_rate

        input_dim = 2 * config.n_harmonics + 2

        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        # Learnable output scale - starts small to avoid dominating the latent
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    @classmethod
    def from_config(cls, config: str, **overrides) -> "F0ConditioningEmbedding":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = F0ConditioningEmbedding.from_config("small", encoder_dim=256)
        """
        if isinstance(config, str):
            if config not in F0_CONDITIONING_EMBEDDING_CONFIGS:
                raise ValueError(f"Unknown config: {config}. Available: {list(F0_CONDITIONING_EMBEDDING_CONFIGS.keys())}")
            config = F0_CONDITIONING_EMBEDDING_CONFIGS[config]

        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = F0ConditioningEmbeddingConfig(**config_dict)

        return cls(config)

    def forward(
        self,
        log_f0: torch.Tensor,
        voiced_prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            log_f0: [B, T] - log fundamental frequency (natural log of Hz)
            voiced_prob: [B, T] - voicing probability (0-1)

        Returns:
            embedding: [B, embedding_dim, T]
        """
        B, T = log_f0.shape
        device = log_f0.device

        # Convert log F0 to Hz
        f0_hz = torch.exp(log_f0)  # [B, T]

        # Create time axis: cumulative phase based on F0
        # For frame t, phase = 2π * f0 * t * frame_duration
        # Use cumulative sum for proper phase continuity
        frame_times = torch.arange(T, device=device, dtype=log_f0.dtype) * self.frame_duration
        frame_times = frame_times.unsqueeze(0).expand(B, -1)  # [B, T]

        # Base phase: 2π * f0 * t
        base_phase = 2 * torch.pi * f0_hz * frame_times  # [B, T]

        # Create harmonic sinusoids
        harmonic_features = []
        for h in range(1, self.config.n_harmonics + 1):
            phase = h * base_phase  # [B, T]
            harmonic_features.append(torch.sin(phase))
            harmonic_features.append(torch.cos(phase))

        # Stack harmonics: [B, T, 2*n_harmonics]
        harmonics = torch.stack(harmonic_features, dim=-1)

        # Gate harmonics by voicing probability
        # Unvoiced frames should have no harmonic content
        voicing_gate = voiced_prob.unsqueeze(-1)  # [B, T, 1]
        harmonics = harmonics * voicing_gate

        # Normalize log F0 for the raw feature
        log_f0_norm = (log_f0 - 5.0) / 1.5  # Roughly [-1, 1]

        # Concatenate: harmonics + log_f0 + voiced
        features = torch.cat([
            harmonics,                           # [B, T, 2*n_harmonics]
            log_f0_norm.unsqueeze(-1),          # [B, T, 1]
            voiced_prob.unsqueeze(-1),          # [B, T, 1]
        ], dim=-1)  # [B, T, 2*n_harmonics + 2]

        emb: torch.Tensor = self.proj(features)  # [B, T, embedding_dim]

        # Scale output
        emb = emb * self.output_scale

        return emb.transpose(1, 2)  # [B, embedding_dim, T]


class SMG(nn.Module):
    """
    SMG (SIVE-Mel Generator): deterministic decoder for mel spectrogram generation from
    latent + speaker + F0.

    This is a decoder-only model — the external caller provides the SIVE encoder features
    (e.g. from a frozen SIVE encoder upstream). There is no encoder and no reparameterization
    trick; conditioning on speaker + F0 is what makes this generative.

    Input: [B, latent_dim, T'] latent representation
    Output: [B, n_mels, T] generated mel spectrogram
    """
    def __init__(
        self,
        config: SMGConfig,
        decoder: Union[SMGDecoder2D, SMGDecoder1D],
        f0_predictor: F0Predictor,
        f0_embedding: F0ConditioningEmbedding,
    ):
        super().__init__()

        self.config = config

        self.decoder = decoder

        # F0 conditioning modules
        self.f0_predictor_input = getattr(config, 'f0_predictor_input', 'features')
        self.f0_predictor = f0_predictor
        self.f0_embedding = f0_embedding

        self.gradient_checkpointing = False

    @classmethod
    def from_config(cls, config: Union[str, SMGConfig], **overrides) -> "SMG":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = SMG.from_config("small", latent_dim=32)
        """
        if isinstance(config, str):
            if config not in SMG_CONFIGS:
                raise ValueError(f"Unknown config: {config}. Available: {list(SMG_CONFIGS.keys())}")
            config = SMG_CONFIGS[config]

        # Peel off overrides that target TOP-LEVEL SMGConfig scalar fields (e.g. the
        # loss weights, which SMG.forward reads via self.config.*) and apply them to
        # a COPY of the config here. The sub-config constructors below only accept
        # decoder/F0 fields, so leaving these in `overrides` would TypeError; and
        # dataclasses.replace avoids mutating the shared SMG_CONFIGS singleton.
        import dataclasses as _dc
        _top_scalar_fields = {
            f.name for f in _dc.fields(config)
            if not _dc.is_dataclass(getattr(config, f.name, None))
        }
        _top_overrides = {k: overrides.pop(k) for k in list(overrides) if k in _top_scalar_fields}
        if _top_overrides:
            config = _dc.replace(config, **_top_overrides)

        # hop_length is ONLY a field of the F0 conditioning embedding (it sets the
        # harmonic phase step = hop/sample_rate). Pop it up front so it doesn't reach
        # the decoder or F0-predictor sub-configs (which have no such field), and
        # re-add to the conditioning config below. Driven from --voice_hop_length so
        # the F0-embedding phase tracks the mel/F0 rate (e.g. 320 for 50 Hz ContentVec).
        _f0emb_hop = overrides.pop('hop_length', None)

        # Select decoder type based on config
        if config.decoder_1d_config is not None:
            config_dict = {k: v for k, v in config.decoder_1d_config.__dict__.items()}
            config_dict.update(overrides)
            decoder = SMGDecoder1D.from_config(config.decoder_1d_config, **config_dict)
        else:
            config_dict = {k: v for k, v in config.decoder_config.__dict__.items()}
            config_dict.update(overrides)
            decoder = SMGDecoder2D.from_config(config.decoder_config, **config_dict)

        # The F0 predictor consumes the SAME SIVE features as the decoder (via its
        # sive_proj), so its encoder_dim must track sive_encoder_dim — not
        # F0PredictorConfig's own default. Capture the effective value (CLI override
        # if given, else the decoder config's) before popping.
        active_decoder_cfg = config.decoder_1d_config if config.decoder_1d_config is not None else config.decoder_config
        sive_encoder_dim = overrides.pop('sive_encoder_dim', active_decoder_cfg.sive_encoder_dim)
        overrides.pop('activation', None)
        # Decoder-only override: the F0 predictor / conditioning sub-configs don't
        # have this field, so drop it before splatting overrides into them.
        overrides.pop('input_instance_norm', None)

        config_dict = {k: v for k, v in config.f0_predictor_config.__dict__.items()}
        config_dict.update(overrides)
        # A contour is 1-dim; features are sive_encoder_dim. This sizes the predictor's
        # sive_proj Conv1d, so it must match what forward() actually feeds it.
        # Set AFTER the overrides splat so a CLI --sive_encoder_dim cannot clobber the 1
        # and hand the F0 head content.
        _f0_in = getattr(config, 'f0_predictor_input', 'features')
        config_dict['encoder_dim'] = 1 if _f0_in == 'contour' else sive_encoder_dim
        # On the contour path the F0 head sees only 1 channel of pitch, so voicing has no
        # content to read unless it gets its own branch. On the legacy features path the
        # shared trunk already sees content, so leave voicing where it is.
        config_dict['vuv_encoder_dim'] = sive_encoder_dim if _f0_in == 'contour' else None
        f0_predictor = F0Predictor.from_config(config.f0_predictor_config, **config_dict)
        config_dict = {k: v for k, v in config.f0_conditioning_embedding_config.__dict__.items()}
        config_dict.update(overrides)
        if _f0emb_hop is not None:
            config_dict['hop_length'] = _f0emb_hop
        f0_embedding = F0ConditioningEmbedding.from_config(config.f0_conditioning_embedding_config, **config_dict)

        return cls(config, decoder, f0_predictor, f0_embedding)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
        if hasattr(self.decoder, 'gradient_checkpointing_enable'):
            self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        if hasattr(self.decoder, 'gradient_checkpointing_disable'):
            self.decoder.gradient_checkpointing_disable()

    def _f0_predictor_input(self, features, f0_contour):
        """What the F0 predictor reads, per config.

        "contour": the world model supplies a speaker-normalized contour and this predictor
        denormalizes it with ECAPA. "features": the classic voice-cloning path, where the
        content features already carry prosody.
        """
        if self.f0_predictor_input == "contour":
            if f0_contour is None:
                # Loudly. Falling back to no F0 conditioning would produce flat delivery
                # that looks like a modelling failure rather than a missing argument --
                # and this SMG is trained on prosody-free units, so F0 is the ONLY place
                # prosody can come from. A caller that forgot it wants to know now.
                raise ValueError(
                    "f0_predictor_input='contour' but no f0_contour was passed. This SMG "
                    "reads a speaker-normalized contour (from the world model's F0 head), "
                    "not content features — its input carries no prosody. Pass f0_contour=, "
                    "or f0_embedding= to bypass the predictor entirely."
                )
            return f0_contour.unsqueeze(1) if f0_contour.dim() == 2 else f0_contour  # (B, 1, T)
        return features

    def decode(self, z, speaker_embedding=None, f0_embedding=None, features=None, f0_contour=None, return_film_stats=False) -> torch.Tensor:
        """
        Decode latent to mel spectrogram.

        Args:
            z: Latent tensor
            speaker_embedding: Speaker embedding for FiLM conditioning
            f0_embedding: Pre-computed F0 embedding (optional)
            features: SIVE features for F0 prediction (optional, used if f0_embedding not provided)
            return_film_stats: Whether to return FiLM statistics
        # Predict F0 if conditioning is enabled but embedding not provided
        """
        if f0_embedding is None:
            f0_src = self._f0_predictor_input(features, f0_contour)
            if f0_src is not None:
                # z IS the content latent, so it stands in when a caller passes only z.
                log_f0_pred, voiced_pred = self.f0_predictor(
                    speaker_embedding, f0_src, content_features=features if features is not None else z)
                f0_embedding = self.f0_embedding(log_f0_pred, voiced_pred)
        return self.decoder(z, speaker_embedding=speaker_embedding, f0_embedding=f0_embedding, return_film_stats=return_film_stats)

    def forward(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        speaker_embedding: torch.Tensor,
        # F0 supervision (optional, for training F0 predictor)
        target_f0: Optional[torch.Tensor] = None,  # [B, T] ground truth log F0
        target_voiced: Optional[torch.Tensor] = None,  # [B, T] ground truth voicing (0 or 1)
        # F0 warmup: use GT F0 instead of predicted F0 for decoder conditioning
        f0_contour: Optional[torch.Tensor] = None,  # speaker-normalized contour; read when f0_predictor_input='contour'
        use_gt_f0: bool = False,  # If True, use target_f0/target_voiced for decoder conditioning
        return_film_stats: bool = False,
    ):
        """
        Forward pass through SMG.

        Args:
            features: [B, D, T'] SIVE features (channel-first)
            target: [B, n_mels, T] Mel-spectrogram features
            mask: [B, T] optional mask for valid frames
            speaker_embedding: [B, speaker_dim] speaker embedding for FiLM conditioning
            target_f0: [B, T] ground truth log F0 for F0 prediction loss (optional)
            target_voiced: [B, T] ground truth voicing mask for VUV loss (optional)
            use_gt_f0: If True, use GT F0 for decoder conditioning instead of predicted F0.
                       F0 predictor is still run for loss computation, but decoder gets GT signal.
                       Useful for warmup to let F0 embedding learn with clean signal.
            return_film_stats: If True, return FiLM layer statistics

        Returns:
            recon_x: Generated mel spectrogram
            losses: Dict with loss components
        """
        z = features

        # F0 prediction and embedding (if enabled)
        # Always predict F0 (for loss computation even during GT warmup)
        log_f0_pred, voiced_pred = self.f0_predictor(
            speaker_embedding, self._f0_predictor_input(features, f0_contour),
            content_features=features)

        # Determine which F0 to use for decoder conditioning
        # Determine which F0 to use for decoder conditioning
        if use_gt_f0 and target_f0 is not None and target_voiced is not None:
            # GT warmup: use ground truth F0 for decoder conditioning
            # This lets the F0 embedding learn with clean signal
            # Downsample GT F0 to match feature resolution if needed
            gt_f0_for_emb = target_f0
            gt_voiced_for_emb = target_voiced
            if gt_f0_for_emb.size(-1) != log_f0_pred.size(-1):
                gt_f0_for_emb = F.interpolate(
                    gt_f0_for_emb.unsqueeze(1), size=log_f0_pred.size(-1), mode='linear', align_corners=False
                ).squeeze(1)
                gt_voiced_for_emb = F.interpolate(
                    gt_voiced_for_emb.unsqueeze(1), size=log_f0_pred.size(-1), mode='linear', align_corners=False
                ).squeeze(1)
            f0_emb = self.f0_embedding(gt_f0_for_emb, gt_voiced_for_emb)
        else:
            # Normal mode: use predicted F0 for decoder conditioning
            f0_emb = self.f0_embedding(log_f0_pred, voiced_pred)

        film_stats = None
        decode_result = self.decode(z, speaker_embedding=speaker_embedding, f0_embedding=f0_emb, return_film_stats=return_film_stats)
        if return_film_stats and isinstance(decode_result, tuple):
            recon_x, film_stats = decode_result
        else:
            recon_x = decode_result

        # Handle 2D decoder output: [B, 1, 80, T] -> [B, 80, T]
        # The 2D decoder returns 4D with channel dim=1, squeeze it for consistency
        if recon_x.dim() == 4 and recon_x.shape[1] == 1:
            recon_x = recon_x.squeeze(1)

        # recon_x is mel specs, not a reconstruction

        # Align reconstruction to input size (decoder stride may cause size mismatch)
        if recon_x.shape != target.shape:
            # Truncate or pad to match input dimensions
            slices = [slice(None)] * recon_x.dim()
            for dim in range(2, recon_x.dim()):  # Skip batch and channel dims
                if recon_x.shape[dim] > target.shape[dim]:
                    slices[dim] = slice(0, target.shape[dim])
                elif recon_x.shape[dim] < target.shape[dim]:
                    # Pad if reconstruction is smaller (shouldn't happen normally)
                    pad_size = target.shape[dim] - recon_x.shape[dim]
                    pad_dims = [0, 0] * (recon_x.dim() - dim - 1) + [0, pad_size]
                    recon_x = F.pad(recon_x, pad_dims)
            recon_x = recon_x[tuple(slices)]

        # Reconstruction losses (with optional masking)
        # Expand mask to match input shape: [B, T] -> [B, 1, 1, T] for 4D input
        # or [B, 1, T] for 3D input
        if target.dim() == 4:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        else:
            mask_expanded = mask.unsqueeze(1)  # [B, 1, T]

        # Masked MSE loss: only compute on valid positions
        squared_error = (recon_x - target) ** 2
        masked_squared_error = squared_error * mask_expanded
        # Sum over all dims except batch, then divide by number of valid elements per sample
        valid_elements = mask_expanded.sum(dim=list(range(1, mask_expanded.dim())), keepdim=False) * target.shape[1]
        if target.dim() == 4:
            valid_elements = valid_elements * target.shape[2]  # Account for H dimension
        mse_loss = (masked_squared_error.sum(dim=list(range(1, masked_squared_error.dim()))) / (valid_elements + 1e-5)).mean()

        # Masked L1 loss
        if self.config.l1_loss_weight > 0:
            abs_error = torch.abs(recon_x - target)
            masked_abs_error = abs_error * mask_expanded
            l1_loss = (masked_abs_error.sum(dim=list(range(1, masked_abs_error.dim()))) / (valid_elements + 1e-5)).mean()
        else:
            l1_loss = torch.tensor(0.0, device=target.device)

        recon_loss = self.config.mse_loss_weight * mse_loss + self.config.l1_loss_weight * l1_loss

        total_loss = (
            self.config.recon_loss_weight * recon_loss
        )

        # F0 prediction losses (if F0 conditioning is enabled and ground truth provided)
        f0_loss = torch.tensor(0.0, device=target.device)
        vuv_loss = torch.tensor(0.0, device=target.device)
        if log_f0_pred is not None and target_f0 is not None and target_voiced is not None:
            # Upsample predictions to match target resolution if needed
            if log_f0_pred.size(-1) != target_f0.size(-1):
                log_f0_pred_up: torch.Tensor = F.interpolate(
                    log_f0_pred.unsqueeze(1), size=target_f0.size(-1), mode='linear', align_corners=False
                ).squeeze(1)
                voiced_pred_up: torch.Tensor = F.interpolate(
                    voiced_pred.unsqueeze(1), size=target_f0.size(-1), mode='linear', align_corners=False
                ).squeeze(1)
            else:
                log_f0_pred_up: torch.Tensor = log_f0_pred
                voiced_pred_up: torch.Tensor = voiced_pred

            # F0 loss weighted by soft voicing probability (target_voiced is 0-1 periodicity)
            # This gives more weight to clearly voiced frames and less to ambiguous ones
            f0_error = torch.abs(log_f0_pred_up - target_f0)
            weighted_f0_error = f0_error * target_voiced  # Weight by voicing confidence
            # Normalize by sum of weights to avoid scale issues
            voicing_sum = target_voiced.sum() + 1e-8
            f0_loss = weighted_f0_error.sum() / voicing_sum

            # Voicing prediction loss (BCE with soft targets)
            # target_voiced is now a soft probability (0-1) from periodicity
            # Use bce_with_logits for numerical stability (avoids NaN from sigmoid + BCE)
            with torch.autocast(device_type='cuda', enabled=False):
                voiced_logit = torch.logit(voiced_pred_up.float(), eps=1e-6)
                vuv_loss = F.binary_cross_entropy_with_logits(
                    voiced_logit,
                    target_voiced.clamp(0, 1).float(),
                    reduction='mean'
                )

            # Add F0 losses to total
            total_loss = total_loss + self.config.f0_loss_weight * f0_loss + self.config.vuv_loss_weight * vuv_loss

        losses = {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "f0_loss": f0_loss,
            "vuv_loss": vuv_loss,
        }

        # Add F0 predictions for external logging/monitoring
        losses["log_f0_pred"] = log_f0_pred
        losses["voiced_pred"] = voiced_pred

        # Add FiLM statistics if requested
        if return_film_stats and film_stats is not None:
            losses["film_stats"] = film_stats

        return recon_x, losses
