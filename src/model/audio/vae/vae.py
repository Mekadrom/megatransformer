from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.audio.vae.vae import AUDIO_VAE_CONFIGS, AUDIO_DECODER_CONFIGS, AUDIO_ENCODER_CONFIGS, F0_CONDITIONING_EMBEDDING_CONFIGS, F0_PREDICTOR_CONFIGS, AudioVAEConfig, F0ConditioningEmbeddingConfig, F0PredictorConfig
from config.audio.vae.vae import AudioVAEDecoderConfig, AudioVAEEncoderConfig
from model import activations
from model.activations import get_activation_type
from model.audio.vae.criteria import AudioPerceptualLoss
from model.audio.vae.residual_block import ResidualBlock1d, ResidualBlock2d


class AudioVAEEncoder(nn.Module):
    """
    VAE encoder for SIVE features.

    Takes SIVE features [B, T, D] and compresses to latent space.
    Uses Conv1D for temporal downsampling.

    Architecture:
    - Input projection: D -> intermediate_channels[0]
    - Downsampling stages with strided Conv1D
    - Optional residual blocks per stage
    - Output: mu, logvar [B, latent_dim, T']
    """
    def __init__(self, config: AudioVAEEncoderConfig):
        super().__init__()

        self.config = config

        self.strides = config.strides
        channels = [config.encoder_dim] + config.intermediate_channels

        # Get activation type
        if config.activation == "snake":
            self.get_activation = lambda c: activations.Snake(c)
        else:
            activation_type = get_activation_type(config.activation)
            if activation_type in [activations.SwiGLU, activations.Snake]:
                self.get_activation = lambda c: activation_type(c)
            else:
                self.get_activation = lambda c: activation_type()

        # Build encoder stages
        self.stages = nn.ModuleList()
        for i, (in_c, out_c, kernel_size, stride) in enumerate(
            zip(channels[:-1], channels[1:], config.kernel_sizes, config.strides)
        ):
            stage = nn.ModuleList()

            # Strided conv for downsampling
            padding = kernel_size // 2
            stage.append(nn.Conv1d(in_c, out_c, kernel_size, stride=stride, padding=padding))
            stage.append(nn.GroupNorm(max(1, out_c // 4), out_c))
            stage.append(self.get_activation(out_c))

            # Residual blocks
            for _ in range(config.n_residual_blocks):
                stage.append(ResidualBlock1d(out_c, out_c, kernel_size=kernel_size, activation_fn=config.activation))

            if config.dropout > 0:
                stage.append(nn.Dropout(config.dropout))

            self.stages.append(stage)

        # Output projections for mu and logvar
        final_channels = config.intermediate_channels[-1]
        self.fc_mu = nn.Conv1d(final_channels, config.latent_channels, kernel_size=3, padding=1)
        self.fc_logvar = nn.Conv1d(final_channels, config.latent_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) and module not in [self.fc_mu, self.fc_logvar]:
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize mu/logvar with smaller variance
        nn.init.normal_(self.fc_mu.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_logvar.bias)

    @classmethod
    def from_config(cls, config: Union[str, AudioVAEEncoderConfig], **overrides) -> "AudioVAEEncoder":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = AudioVAEEncoder.from_config("small", latent_dim=32)
        """
        if isinstance(config, str):
            if config not in AUDIO_ENCODER_CONFIGS:
                raise ValueError(f"Unknown config: {config}. Available: {list(AUDIO_ENCODER_CONFIGS.keys())}")
            config = AUDIO_ENCODER_CONFIGS[config]

        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = AudioVAEEncoderConfig(**config_dict)

        return cls(config)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [B, T, D] SIVE features

        Returns:
            mu: [B, latent_dim, T'] latent mean
            logvar: [B, latent_dim, T'] latent log variance
        """
        # Process through encoder stages
        for stage in self.stages:
            for layer in stage:
                x = layer(x)

        # Get mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10.0, max=self.config.logvar_clamp_max)

        return mu, logvar


class AudioVAEDecoder(nn.Module):
    """
    Audio VAE decoder that transitions from 1D (SIVE features latents) to 2D convolutions (mel specs).

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
    def __init__(self, config: AudioVAEDecoderConfig):
        super().__init__()

        self.config = config

        # Effective speaker dim for FiLM
        self.film_speaker_dim = config.speaker_embedding_proj_dim if config.speaker_embedding_proj_dim > 0 else config.speaker_embedding_dim

        # Input dim is just latent (F0 is injected after 1D→2D transition)
        decoder_input_dim = config.latent_channels

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
    def from_config(cls, config: Union[str, AudioVAEDecoderConfig], **overrides) -> "AudioVAEDecoder":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = AudioVAEDecoder.from_config("small", latent_dim=32)
        """
        if isinstance(config, str):
            if config not in AUDIO_DECODER_CONFIGS:
                raise ValueError(f"Unknown config: {config}. Available: {list(AUDIO_DECODER_CONFIGS.keys())}")
            config = AUDIO_DECODER_CONFIGS[config]

        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = AudioVAEDecoderConfig(**config_dict)

        return cls(config)

    def get_output_length(self, input_length: int) -> int:
        """Compute output temporal length given latent length."""
        length = input_length
        if self.config.conv1d_upsample_factor > 1:
            length = length * self.config.conv1d_upsample_factor
        for scale_factor in self.config.scale_factors_2d:
            length = length * scale_factor[1]  # Time dimension
        return length

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
            for layer in stage:
                x = layer(x)

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


class F0Predictor(nn.Module):
    """
    Predicts F0 (fundamental frequency) contour from speaker embedding + SIVE features.

    Speaker embedding provides F0 range/characteristics (who is speaking).
    SIVE features provide prosodic timing cues (where stress/emphasis occurs).

    Output is per-frame: log F0 values and voiced/unvoiced probability.
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

        # Output: log_f0 (continuous) + voiced_logit
        self.output_proj = nn.Conv1d(config.hidden_dim, 2, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # Initialize output layer to produce reasonable F0 values
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            # Initialize log_f0 bias to ~5.0 (≈150 Hz), voiced bias to 0
            self.output_proj.bias.data[0] = 5.0  # log(150) ≈ 5.0
            self.output_proj.bias.data[1] = 0.0

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            speaker_embedding: [B, speaker_dim] or [B, 1, speaker_dim]
            sive_features: [B, sive_dim, T']

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
        out = self.output_proj(x)  # [B, 2, T']
        log_f0 = out[:, 0, :]      # [B, T']
        voiced_prob = torch.sigmoid(out[:, 1, :])  # [B, T']

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


class AudioVAE(nn.Module):
    """
    Complete VAE for SIVE features -> Mel Specs.

    Compresses speaker-invariant speech features to a lower-dimensional
    latent space while preserving linguistic/prosodic structure.

    Input: [B, T, D] SIVE features
    Latent: [B, latent_dim, T'] compressed representation
    Output: [B, T, D] reconstructed features
    """
    def __init__(
        self,
        config: AudioVAEConfig,
        encoder: AudioVAEEncoder,
        decoder: AudioVAEDecoder,
        f0_predictor: F0Predictor,
        f0_embedding: F0ConditioningEmbedding,
    ):
        super().__init__()

        self.config = config

        self.encoder = encoder
        self.decoder = decoder

        # F0 conditioning modules (optional)
        self.f0_predictor = f0_predictor
        self.f0_embedding = f0_embedding

        if config.multi_scale_mel_weight > 0:
            self.perceptual_loss = AudioPerceptualLoss(multi_scale_mel_weight=config.multi_scale_mel_weight)

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "AudioVAE":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = AudioVAE.from_config("small", latent_dim=32)
        """
        if config_name not in AUDIO_VAE_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(AUDIO_VAE_CONFIGS.keys())}")
        
        config = AUDIO_VAE_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = AudioVAEConfig(**config_dict)

        config_dict = {k: v for k, v in config.encoder_config.__dict__.items()}
        config_dict.update(overrides)
        encoder = AudioVAEEncoder.from_config(config.encoder_config, **config_dict)

        config_dict = {k: v for k, v in config.decoder_config.__dict__.items()}
        config_dict.update(overrides)
        decoder = AudioVAEDecoder.from_config(config.decoder_config, **config_dict)

        config_dict = {k: v for k, v in config.f0_predictor_config.__dict__.items()}
        config_dict.update(overrides)
        f0_predictor = F0Predictor.from_config(config.f0_predictor_config, **config_dict)
        config_dict = {k: v for k, v in config.f0_conditioning_embedding_config.__dict__.items()}
        config_dict.update(overrides)
        f0_embedding = F0ConditioningEmbedding.from_config(config.f0_conditioning_embedding_config, **config_dict)

        return cls(config, encoder, decoder, f0_predictor, f0_embedding)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.

        Returns:
            mu: Latent mean
            logvar: Latent log variance
            learned_speaker_emb (optional): [B, learned_speaker_dim] if encoder.learn_speaker_embedding=True
        """
        return self.encoder(x)

    def decode(self, z, speaker_embedding=None, f0_embedding=None, features=None, return_film_stats=False) -> torch.Tensor:
        """
        Decode latent to mel spectrogram.

        Args:
            z: Latent tensor
            speaker_embedding: Speaker embedding for FiLM conditioning
            f0_embedding: Pre-computed F0 embedding (optional)
            features: SIVE features for F0 prediction (optional, used if f0_embedding not provided)
            return_film_stats: Whether to return FiLM statistics

        If F0 conditioning is enabled and f0_embedding is not provided but features are,
        F0 will be predicted automatically from speaker_embedding + features.
        """
        # Predict F0 if conditioning is enabled but embedding not provided
        log_f0_pred, voiced_pred = self.f0_predictor(speaker_embedding, features)
        f0_embedding = self.f0_embedding(log_f0_pred, voiced_pred)
        return self.decoder(z, speaker_embedding=speaker_embedding, f0_embedding=f0_embedding, return_film_stats=return_film_stats)

    def forward(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        speaker_embedding: torch.Tensor,
        kl_weight_multiplier: float = 1.0,
        # F0 supervision (optional, for training F0 predictor)
        target_f0: Optional[torch.Tensor] = None,  # [B, T] ground truth log F0
        target_voiced: Optional[torch.Tensor] = None,  # [B, T] ground truth voicing (0 or 1)
        # F0 warmup: use GT F0 instead of predicted F0 for decoder conditioning
        use_gt_f0: bool = False,  # If True, use target_f0/target_voiced for decoder conditioning
        return_film_stats: bool = False,
    ):
        """
        Forward pass through VAE.

        Args:
            features: [B, D, T'] SIVE features (channel-first)
            target: [B, n_mels, T] Mel-spectrogram features
            mask: [B, T] optional mask for valid frames
            speaker_embedding: [B, speaker_dim] speaker embedding for FiLM conditioning
            kl_weight_multiplier: Multiplier for KL loss (for annealing)
            target_f0: [B, T] ground truth log F0 for F0 prediction loss (optional)
            target_voiced: [B, T] ground truth voicing mask for VUV loss (optional)
            use_gt_f0: If True, use GT F0 for decoder conditioning instead of predicted F0.
                       F0 predictor is still run for loss computation, but decoder gets GT signal.
                       Useful for warmup to let F0 embedding learn with clean signal.
            return_film_stats: If True, return FiLM layer statistics

        Returns:
            recon_x: Reconstructed mel spectrogram
            mu: Latent mean
            logvar: Latent log variance
            losses: Dict with loss components
        """
        mu, logvar = self.encode(features)

        z = self.reparameterize(mu, logvar)

        # F0 prediction and embedding (if enabled)
        # Always predict F0 (for loss computation even during GT warmup)
        log_f0_pred, voiced_pred = self.f0_predictor(speaker_embedding, features)

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

        # Align reconstruction to input size (encoder-decoder stride may cause size mismatch)
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

        # Compute KL divergence with optional free bits
        # Per-element KL: [B, C, H, W] for images, [B, C, T] for audio
        kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if self.config.free_bits > 0:
            # Free bits: apply minimum KL per channel to prevent posterior collapse
            # Sum over spatial dims, mean over batch -> per-channel KL: [C]
            spatial_dims = list(range(2, mu.dim()))  # [2, 3] for 4D, [2] for 3D
            kl_per_channel = kl_per_element.sum(dim=spatial_dims).mean(dim=0)  # [C]

            # Clamp each channel's KL to at least free_bits
            kl_per_channel = torch.clamp(kl_per_channel, min=self.config.free_bits)

            # Sum over channels for total KL
            kl_divergence = kl_per_channel.sum()
        else:
            # Original behavior: sum over all latent dims, mean over batch
            latent_dims = list(range(1, mu.dim()))  # [1, 2, 3] for 4D, [1, 2] for 3D
            kl_divergence = kl_per_element.sum(dim=latent_dims).mean()

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

        perceptual_loss_value = torch.tensor(0.0, device=target.device)
        # Perceptual loss (if enabled)
        if hasattr(self, 'perceptual_loss'):
            perceptual_loss_value = self.perceptual_loss(recon_x, target, mask=mask)["total_perceptual_loss"]

        recon_loss = self.config.mse_loss_weight * mse_loss + self.config.l1_loss_weight * l1_loss

        # Apply KL weight multiplier for KL annealing
        effective_kl_weight = self.config.kl_divergence_loss_weight * kl_weight_multiplier

        total_loss = (
            self.config.recon_loss_weight * recon_loss
            + effective_kl_weight * kl_divergence
            + self.config.perceptual_loss_weight * perceptual_loss_value
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
            # Disable autocast for BCE (not autocast-safe with post-sigmoid values)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                vuv_loss = F.binary_cross_entropy(
                    voiced_pred_up.clamp(1e-7, 1 - 1e-7).float(),
                    target_voiced.clamp(0, 1).float(),  # Ensure valid probability range
                    reduction='mean'
                )

            # Add F0 losses to total
            total_loss = total_loss + self.config.f0_loss_weight * f0_loss + self.config.vuv_loss_weight * vuv_loss

        losses = {
            "total_loss": total_loss,
            "kl_divergence": kl_divergence,
            "recon_loss": recon_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "f0_loss": f0_loss,
            "vuv_loss": vuv_loss,
            "kl_weight_multiplier": torch.tensor(kl_weight_multiplier, device=target.device),
        }

        # Add F0 predictions for external logging/monitoring
        losses["log_f0_pred"] = log_f0_pred
        losses["voiced_pred"] = voiced_pred

        # Add FiLM statistics if requested
        if return_film_stats and film_stats is not None:
            losses["film_stats"] = film_stats

        return recon_x, mu, logvar, losses

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class AudioCVAEDecoderOnly(nn.Module):
    """
    CVAE Decoder-only model for mel spectrogram generation from latent + speaker + F0.

    This model uses only the decoder part of the AudioVAE, allowing it to be
    used in scenarios where the latent representation is provided externally,
    such as in a conditional generation setup.

    Input: [B, latent_dim, T'] latent representation
    Output: [B, 1, n_mels, T] reconstructed mel spectrogram
    """
    def __init__(
        self,
        config: AudioVAEDecoderConfig,
        decoder: AudioVAEDecoder,
        f0_predictor: F0Predictor,
        f0_embedding: F0ConditioningEmbedding,
    ):
        super().__init__()

        self.config = config

        self.decoder = decoder

        # F0 conditioning modules
        self.f0_predictor = f0_predictor
        self.f0_embedding = f0_embedding

    @classmethod
    def from_config(cls, config: Union[str, AudioVAEConfig], **overrides) -> "AudioCVAEDecoderOnly":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = AudioCVAEDecoderOnly.from_config("small", latent_dim=32)
        """
        if isinstance(config, str):
            if config not in AUDIO_VAE_CONFIGS:
                raise ValueError(f"Unknown config: {config}. Available: {list(AUDIO_VAE_CONFIGS.keys())}")
            config = AUDIO_VAE_CONFIGS[config]

        # Apply overrides
        config_dict = {k: v for k, v in config.decoder_config.__dict__.items()}
        config_dict.update(overrides)
        decoder = AudioVAEDecoder.from_config(config.decoder_config, **config_dict)

        config_dict = {k: v for k, v in config.f0_predictor_config.__dict__.items()}
        config_dict.update(overrides)
        f0_predictor = F0Predictor.from_config(config.f0_predictor_config, **config_dict)
        config_dict = {k: v for k, v in config.f0_conditioning_embedding_config.__dict__.items()}
        config_dict.update(overrides)
        f0_embedding = F0ConditioningEmbedding.from_config(config.f0_conditioning_embedding_config, **config_dict)

        return cls(config.decoder_config, decoder, f0_predictor, f0_embedding)
    
    def decode(self, z, speaker_embedding=None, f0_embedding=None, features=None, return_film_stats=False) -> torch.Tensor:
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
        if f0_embedding is None and features is not None:
            log_f0_pred, voiced_pred = self.f0_predictor(speaker_embedding, features)
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
        use_gt_f0: bool = False,  # If True, use target_f0/target_voiced for decoder conditioning
        return_film_stats: bool = False,
    ):
        """
        Forward pass through VAE.

        Args:
            features: [B, D, T'] SIVE features (channel-first)
            target: [B, n_mels, T] Mel-spectrogram features
            mask: [B, T] optional mask for valid frames
            speaker_embedding: [B, speaker_dim] speaker embedding for FiLM conditioning
            kl_weight_multiplier: Multiplier for KL loss (for annealing)
            target_f0: [B, T] ground truth log F0 for F0 prediction loss (optional)
            target_voiced: [B, T] ground truth voicing mask for VUV loss (optional)
            use_gt_f0: If True, use GT F0 for decoder conditioning instead of predicted F0.
                       F0 predictor is still run for loss computation, but decoder gets GT signal.
                       Useful for warmup to let F0 embedding learn with clean signal.
            return_film_stats: If True, return FiLM layer statistics

        Returns:
            recon_x: Reconstructed mel spectrogram
            mu: Latent mean
            logvar: Latent log variance
            losses: Dict with loss components
        """
        z = features

        # F0 prediction and embedding (if enabled)
        # Always predict F0 (for loss computation even during GT warmup)
        log_f0_pred, voiced_pred = self.f0_predictor(speaker_embedding, features)

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

        # Align reconstruction to input size (encoder-decoder stride may cause size mismatch)
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

        perceptual_loss_value = torch.tensor(0.0, device=target.device)
        # Perceptual loss (if enabled)
        if hasattr(self, 'perceptual_loss'):
            perceptual_loss_value = self.perceptual_loss(recon_x, target, mask=mask)["total_perceptual_loss"]

        recon_loss = self.config.mse_loss_weight * mse_loss + self.config.l1_loss_weight * l1_loss

        total_loss = (
            self.config.recon_loss_weight * recon_loss
            + self.config.perceptual_loss_weight * perceptual_loss_value
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
            # Disable autocast for BCE (not autocast-safe with post-sigmoid values)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                vuv_loss = F.binary_cross_entropy(
                    voiced_pred_up.clamp(1e-7, 1 - 1e-7).float(),
                    target_voiced.clamp(0, 1).float(),  # Ensure valid probability range
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
