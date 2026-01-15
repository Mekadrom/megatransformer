import torch
import torch.nn as nn
import torch.nn.functional as F

from model import activations
from model.audio.attention import AudioConvSelfAttentionBlock, AudioConv2DSelfAttentionBlock
from model.vae import VAE
from utils import megatransformer_utils
from utils.model_utils import get_activation_type


class Snake2d(nn.Module):
    """
    Snake activation for 2D inputs: x + (1/alpha) * sin^2(alpha * x)

    The alpha parameter is learnable per channel, allowing the network
    to adapt the periodicity to different frequency ranges.
    Designed for audio spectrograms where periodic structure is important.
    """
    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        # Learnable frequency parameter per channel
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2


class ResidualBlock2d(nn.Module):
    """
    Residual block for audio VAE encoder/decoder.

    Uses two conv layers with GroupNorm and a skip connection.
    Optionally supports channel changes via a 1x1 conv projection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel_size: tuple = (3, 5),
        activation_fn: str = "silu",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # Get activation
        if activation_fn == "snake":
            self.act1 = Snake2d(out_channels)
            self.act2 = Snake2d(out_channels)
        else:
            activation_type = get_activation_type(activation_fn)
            if activation_type in [activations.SwiGLU, activations.Snake]:
                self.act1 = activation_type(out_channels)
                self.act2 = activation_type(out_channels)
            else:
                self.act1 = activation_type()
                self.act2 = activation_type()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(max(1, out_channels // 4), out_channels)
        self.norm2 = nn.GroupNorm(max(1, out_channels // 4), out_channels)

        # Skip connection projection if channels change
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for module in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize second conv with smaller weights for stable residual learning
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.conv2.weight.data *= 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip_proj is None else self.skip_proj(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.act2(x)

        return x


class BottleneckAttention(nn.Module):
    """
    Full 2D self-attention module for the VAE bottleneck with 2D RoPE.

    Uses AudioConv2DSelfAttentionBlock for full attention over the M×T
    frequency-time grid, enabling:
    - Cross-frequency attention (harmonics, formant relationships)
    - Cross-time attention (temporal dependencies)
    - 2D positional encoding via RoPE

    For a bottleneck with M=10, T=75, this creates a 750-token sequence.
    Supports padding masks to prevent attention to padded positions.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_freq_positions: int = 128,
        max_time_positions: int = 512,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Ensure head dimension is divisible by 4 for 2D RoPE
        head_dim = channels // num_heads
        if head_dim % 4 != 0:
            # Adjust num_heads to ensure head_dim is divisible by 4
            # Find largest num_heads where channels // num_heads is divisible by 4
            for n in range(num_heads, 0, -1):
                if (channels // n) % 4 == 0 and channels % n == 0:
                    num_heads = n
                    head_dim = channels // n
                    break
            else:
                raise ValueError(
                    f"Cannot find valid num_heads for channels={channels} where "
                    f"head_dim is divisible by 4. Consider using channels divisible by 16."
                )
            self.num_heads = num_heads

        print(f"BottleneckAttention: adjusted num_heads={self.num_heads}, head_dim={head_dim} for 2D RoPE")

        # pre-norm
        self.norm = nn.GroupNorm(max(1, channels // 4), channels)

        # Use 2D attention with RoPE
        self.attention = AudioConv2DSelfAttentionBlock(
            hidden_size=channels,
            n_heads=num_heads,
            d_queries=head_dim,
            d_values=head_dim,
            max_freq_positions=max_freq_positions,
            max_time_positions=max_time_positions,
            kernel_size=3,
            use_depthwise=True,
            use_flash_attention=True,
            dropout_p=dropout,
        )

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # Initialize output projection with small weights for stable training
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, M, T] input tensor
            lengths: [B] optional tensor of valid time lengths (in T dimension).
                     If provided, positions beyond each sample's length are masked
                     for all frequency bins.
            return_attention_weights: If True, also return attention weights.
        Returns:
            [B, C, M, T] output tensor with 2D self-attention applied
            attn_weights (optional): [B, n_heads, M*T, M*T] if return_attention_weights=True
        """

        B, C, M, T = x.shape
        residual = x

        # Normalize
        x = self.norm(x)

        # Create padding mask if lengths provided
        # key_padding_mask: [B, T] where True = masked (the 2D attention block
        # internally expands this to [B, M*T] since each mel bin at time t is masked)
        key_padding_mask = None
        if lengths is not None:
            # [B, T]: True where position should be masked
            key_padding_mask = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
            key_padding_mask = key_padding_mask.expand(B, T)  # [B, T]
            key_padding_mask = key_padding_mask >= lengths.unsqueeze(1)  # [B, T]

        # 2D self-attention with optional masking
        attn_result = self.attention(
            x,
            key_padding_mask=key_padding_mask,
            return_attention_weights=return_attention_weights,
        )

        if return_attention_weights:
            attn_out, attn_weights = attn_result
        else:
            attn_out = attn_result
            attn_weights = None

        # Project and add residual
        out = self.out_proj(attn_out)

        if return_attention_weights and attn_weights is not None:
            return residual + out, attn_weights

        return residual + out


class AudioVAEEncoder(nn.Module):
    """
    Audio VAE encoder with:
    - Residual blocks for better gradient flow
    - Snake activation for audio-specific periodicity
    - Optional bottleneck attention for long-range dependencies
    - Larger receptive fields via configurable kernel sizes

    Supports padding masks via lengths parameter to prevent attention to padded positions.
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        intermediate_channels: list = [64, 128, 256],
        kernel_sizes: list = [(3, 9), (3, 9), (3, 5)],
        strides: list = [(2, 5), (2, 5), (2, 1)],
        n_residual_blocks: int = 2,
        use_attention: bool = True,
        attention_heads: int = 4,
        activation_fn: str = "snake",
        speaker_embedding_dim: int = 0,
        normalize_speaker_embedding: bool = True,
        film_scale_bound: float = 0.5,
        film_shift_bound: float = 0.5,
        logvar_clamp_max: float = 4.0,
        # Learned speaker embedding parameters
        learn_speaker_embedding: bool = False,  # If True, encoder outputs learned speaker embedding
        learned_speaker_dim: int = 256,  # Dimension of learned speaker embedding output
    ):
        super().__init__()
        self.use_attention = use_attention
        self.speaker_embedding_dim = speaker_embedding_dim
        self.normalize_speaker_embedding = normalize_speaker_embedding
        self.film_scale_bound = film_scale_bound
        self.film_shift_bound = film_shift_bound
        self.logvar_clamp_max = logvar_clamp_max
        self.learn_speaker_embedding = learn_speaker_embedding
        self.learned_speaker_dim = learned_speaker_dim

        channels = [in_channels] + intermediate_channels

        # Store time strides for computing downsampled lengths
        self.time_strides = [s[1] for s in strides]

        # Build encoder stages
        self.stages = nn.ModuleList()
        for in_c, out_c, kernel_size, stride in zip(
            channels[:-1], channels[1:], kernel_sizes, strides
        ):
            stage = nn.ModuleList()

            # Strided conv for downsampling
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            stage.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding))
            stage.append(nn.GroupNorm(max(1, out_c // 4), out_c))

            # Activation after downsampling
            if activation_fn == "snake":
                stage.append(Snake2d(out_c))
            else:
                activation_type = get_activation_type(activation_fn)
                if activation_type in [activations.SwiGLU, activations.Snake]:
                    stage.append(activation_type(out_c))
                else:
                    stage.append(activation_type())

            # Residual blocks
            for _ in range(n_residual_blocks):
                stage.append(ResidualBlock2d(out_c, out_c, kernel_size=kernel_size, activation_fn=activation_fn))

            self.stages.append(nn.ModuleList(stage))

        print("Encoder BottleneckAttention: initializing 2D self-attention module with "
                f"channels={intermediate_channels[-1]}, num_heads={attention_heads}")

        # Bottleneck attention (on smallest resolution)
        if use_attention:
            self.attention = BottleneckAttention(
                channels=intermediate_channels[-1],
                num_heads=attention_heads,
            )

        # Speaker embedding projections for FiLM conditioning
        if speaker_embedding_dim > 0:
            self.speaker_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(speaker_embedding_dim, speaker_embedding_dim),
                    nn.SiLU(),
                    nn.Linear(speaker_embedding_dim, out_c * 2),
                )
                for out_c in intermediate_channels
            ])

        # Mu and logvar projections
        self.fc_mu = nn.Conv2d(channels[-1], latent_channels, kernel_size=(3, 5), padding=(1, 2))
        self.fc_logvar = nn.Conv2d(channels[-1], latent_channels, kernel_size=(3, 5), padding=(1, 2))

        # Learned speaker embedding head
        # Uses global average pooling to remove temporal/spatial structure, then MLP to project
        # This forces the head to learn a single speaker vector without temporal info
        if learn_speaker_embedding:
            self.speaker_head = nn.Sequential(
                nn.Linear(intermediate_channels[-1], learned_speaker_dim),
                nn.SiLU(),
                nn.Linear(learned_speaker_dim, learned_speaker_dim),
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and module not in [self.fc_mu, self.fc_logvar]:
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize mu/logvar with smaller variance for stable latent space
        nn.init.normal_(self.fc_mu.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_logvar.bias)

        # Initialize FiLM projections
        if self.speaker_embedding_dim > 0:
            for proj in self.speaker_projections:
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

        # Initialize learned speaker embedding head
        if self.learn_speaker_embedding and hasattr(self, 'speaker_head'):
            for module in self.speaker_head:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        speaker_embedding=None,
        lengths: torch.Tensor = None,
        return_attention_weights: bool = False,
    ) -> tuple:
        """
        Args:
            x: [B, C, H, W] input tensor (mel spectrogram)
            lengths: [B] optional tensor of valid time lengths (in W dimension).
                     If provided, downsampled lengths are computed and passed to attention.
            return_attention_weights: If True, also return attention weights from bottleneck.

        Returns:
            mu: [B, latent_channels, H', W'] latent mean
            logvar: [B, latent_channels, H', W'] latent log variance
            learned_speaker_emb (optional): [B, learned_speaker_dim] if learn_speaker_embedding=True
            attn_weights (optional): [B, M, n_heads, T, T] if return_attention_weights=True and use_attention
        """
        # Process speaker embedding
        if speaker_embedding is not None and self.speaker_embedding_dim > 0:
            if speaker_embedding.dim() == 3:
                speaker_embedding = speaker_embedding.squeeze(1)
            if self.normalize_speaker_embedding:
                speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)

        # Process through encoder stages
        for i, stage in enumerate(self.stages):
            for layer in stage:
                x = layer(x)

            # Apply FiLM conditioning after each stage
            if speaker_embedding is not None and self.speaker_embedding_dim > 0:
                film_params = self.speaker_projections[i](speaker_embedding)
                out_c = x.shape[1]
                scale = film_params[:, :out_c].unsqueeze(-1).unsqueeze(-1)
                shift = film_params[:, out_c:].unsqueeze(-1).unsqueeze(-1)

                if self.film_scale_bound > 0:
                    scale = self.film_scale_bound * torch.tanh(scale)
                if self.film_shift_bound > 0:
                    shift = self.film_shift_bound * torch.tanh(shift)

                x = x * (1 + scale) + shift

        # Compute downsampled lengths for attention mask
        attn_lengths = None
        if lengths is not None and self.use_attention:
            attn_lengths = lengths
            for stride in self.time_strides:
                # Ceiling division: (L + stride - 1) // stride
                attn_lengths = (attn_lengths + stride - 1) // stride

        # Apply bottleneck attention
        attn_weights = None
        if self.use_attention:
            attn_result = self.attention(
                x,
                lengths=attn_lengths,
                return_attention_weights=return_attention_weights,
            )
            if return_attention_weights:
                x, attn_weights = attn_result
            else:
                x = attn_result

        # Compute learned speaker embedding via global average pooling
        # This removes temporal/spatial structure, forcing the head to learn speaker-only info
        learned_speaker_emb = None
        if self.learn_speaker_embedding:
            # x shape: [B, C, M, T] -> global avg pool -> [B, C]
            speaker_features = x.mean(dim=(2, 3))  # [B, C]
            learned_speaker_emb = self.speaker_head(speaker_features)  # [B, learned_speaker_dim]

        # Get mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10.0, max=self.logvar_clamp_max)

        # Build return tuple based on what's enabled
        if self.learn_speaker_embedding:
            if return_attention_weights:
                return mu, logvar, learned_speaker_emb, attn_weights
            return mu, logvar, learned_speaker_emb

        if return_attention_weights:
            return mu, logvar, attn_weights

        return mu, logvar


class AudioVAEDecoder(nn.Module):
    """
    Audio VAE decoder with:
    - Residual blocks for better gradient flow
    - Snake activation for audio-specific periodicity
    - Optional bottleneck attention for long-range dependencies
    - FiLM conditioning for speaker embeddings (early + per-stage)

    Supports padding masks via lengths parameter to prevent attention to padded positions.

    FiLM conditioning is applied:
    1. Early: Right after initial projection (before attention and upsampling)
    2. Per-stage: After each upsampling stage

    This allows speaker characteristics to influence both low-level and high-level features.
    """
    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 1,
        intermediate_channels: list = [256, 128, 64],
        scale_factors: list = [(2, 1), (2, 5), (2, 5)],
        kernel_sizes: list = [(3, 5), (3, 9), (3, 9)],
        n_residual_blocks: int = 2,
        use_attention: bool = True,
        attention_heads: int = 4,
        activation_fn: str = "snake",
        speaker_embedding_dim: int = 0,
        speaker_embedding_proj_dim: int = 0,  # If > 0, project speaker embedding to this dim before FiLM
        normalize_speaker_embedding: bool = True,
        film_scale_bound: float = 0.5,
        film_shift_bound: float = 0.5,
        use_early_film: bool = True,  # Apply FiLM before upsampling
        zero_init_film_bias: bool = False,  # Zero-init ALL FiLM biases to prevent bias-dominated output
        film_no_bias: bool = False,  # Remove bias from FiLM projections entirely (zero emb = zero modulation)
    ):
        super().__init__()

        self.use_attention = use_attention
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_embedding_proj_dim = speaker_embedding_proj_dim
        self.normalize_speaker_embedding = normalize_speaker_embedding
        self.film_scale_bound = film_scale_bound
        self.film_shift_bound = film_shift_bound
        self.use_early_film = use_early_film
        self.zero_init_film_bias = zero_init_film_bias
        self.film_no_bias = film_no_bias

        # Determine the effective speaker embedding dimension for FiLM layers
        # If proj_dim > 0, we project down before FiLM; otherwise use full dim
        self.film_speaker_dim = speaker_embedding_proj_dim if speaker_embedding_proj_dim > 0 else speaker_embedding_dim

        # Speaker embedding projection (bottleneck to reduce params in FiLM layers)
        self.speaker_embedding_projection = None
        if speaker_embedding_dim > 0 and speaker_embedding_proj_dim > 0 and speaker_embedding_proj_dim != speaker_embedding_dim:
            self.speaker_embedding_projection = nn.Sequential(
                nn.Linear(speaker_embedding_dim, speaker_embedding_proj_dim),
                nn.SiLU(),
            )

        # Store time scale factors for reference (if needed for length computations)
        self.time_scale_factors = [s[1] for s in scale_factors]

        # Initial projection from latent space
        self.initial_conv = nn.Conv2d(latent_channels, intermediate_channels[0], kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(max(1, intermediate_channels[0] // 4), intermediate_channels[0])

        if activation_fn == "snake":
            self.initial_act = Snake2d(intermediate_channels[0])
        else:
            activation_type = get_activation_type(activation_fn)
            if activation_type in [activations.SwiGLU, activations.Snake]:
                self.initial_act = activation_type(intermediate_channels[0])
            else:
                self.initial_act = activation_type()

        print("Decoder BottleneckAttention: initializing 2D self-attention module with "
                f"channels={intermediate_channels[0]}, num_heads={attention_heads}")

        # Bottleneck attention (before upsampling, at smallest resolution)
        if use_attention:
            self.attention = BottleneckAttention(
                channels=intermediate_channels[0],
                num_heads=attention_heads,
            )

        # Early FiLM projection (applied before attention/upsampling)
        # This allows speaker info to influence the bottleneck representation
        # Uses film_speaker_dim (projected dim if projection enabled, else full dim)
        if speaker_embedding_dim > 0 and use_early_film:
            self.early_film_projection = nn.Sequential(
                nn.Linear(self.film_speaker_dim, self.film_speaker_dim, bias=not film_no_bias),
                nn.SiLU(),
                nn.Linear(self.film_speaker_dim, intermediate_channels[0] * 2, bias=not film_no_bias),
            )

        # Speaker embedding projections for FiLM conditioning (per-stage)
        # Uses film_speaker_dim (projected dim if projection enabled, else full dim)
        if speaker_embedding_dim > 0:
            self.speaker_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.film_speaker_dim, self.film_speaker_dim, bias=not film_no_bias),
                    nn.SiLU(),
                    nn.Linear(self.film_speaker_dim, out_c * 2, bias=not film_no_bias),
                )
                for out_c in intermediate_channels
            ])

        # Build decoder stages
        self.stages = nn.ModuleList()
        self.activation_fn = activation_fn
        all_channels = [intermediate_channels[0]] + intermediate_channels

        for i, (in_c, out_c, scale_factor, kernel_size) in enumerate(
            zip(all_channels[:-1], intermediate_channels, scale_factors, kernel_sizes)
        ):
            stage = nn.ModuleList()

            # Upsample
            stage.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))

            # Conv after upsample
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            stage.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding))
            stage.append(nn.GroupNorm(max(1, out_c // 4), out_c))

            # Activation
            if activation_fn == "snake":
                stage.append(Snake2d(out_c))
            else:
                activation_type = get_activation_type(activation_fn)
                if activation_type in [activations.SwiGLU, activations.Snake]:
                    stage.append(activation_type(out_c))
                else:
                    stage.append(activation_type())

            # Residual blocks
            for _ in range(n_residual_blocks):
                stage.append(ResidualBlock2d(out_c, out_c, kernel_size=kernel_size, activation_fn=activation_fn))

            self.stages.append(nn.ModuleList(stage))

        # Final output conv
        self.final_conv = nn.Conv2d(intermediate_channels[-1], out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize FiLM projections
        if self.speaker_embedding_dim > 0:
            for proj in self.speaker_projections:
                self._init_film_projection(proj)
            # Also init early_film_projection if it exists
            if hasattr(self, 'early_film_projection'):
                self._init_film_projection(self.early_film_projection)

    def _init_film_projection(self, proj):
        """Initialize a FiLM projection module."""
        if isinstance(proj, nn.Sequential):
            linear_layers = [m for m in proj if isinstance(m, nn.Linear)]
            for i, linear in enumerate(linear_layers):
                if i < len(linear_layers) - 1:
                    # Intermediate layers: default weight init, zero bias
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)
                else:
                    # Output layer
                    if self.zero_init_film_bias:
                        # Zero-init weights AND bias so FiLM starts with NO modulation
                        # This forces the model to learn to use speaker embeddings
                        nn.init.zeros_(linear.weight)
                    else:
                        # Default: small random weights
                        nn.init.normal_(linear.weight, mean=0.0, std=0.02)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)

    def forward(
        self,
        z: torch.Tensor,
        speaker_embedding=None,
        lengths: torch.Tensor = None,
        return_attention_weights: bool = False,
        return_film_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict]:
        """
        Forward pass through decoder.

        Args:
            z: Latent tensor [B, latent_channels, H, W]
            speaker_embedding: Optional speaker embedding [B, 1, speaker_embedding_dim] or [B, speaker_embedding_dim]
            lengths: [B] optional tensor of valid time lengths in latent space (W dimension).
                     If provided, attention masking is applied to padded positions.
            return_attention_weights: If True, also return attention weights from bottleneck.
            return_film_stats: If True, return FiLM statistics dict for monitoring.

        Returns:
            Reconstructed output [B, out_channels, H', W']
            attn_weights (optional): [B, M, n_heads, T, T] if return_attention_weights=True and use_attention
            film_stats (optional): dict with FiLM scale/shift statistics if return_film_stats=True
        """
        # Initialize FiLM statistics tracking
        film_stats = {} if return_film_stats else None

        # Process speaker embedding
        if speaker_embedding is not None and self.speaker_embedding_dim > 0:
            if speaker_embedding.dim() == 3:
                speaker_embedding = speaker_embedding.squeeze(1)
            if self.normalize_speaker_embedding:
                speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)
            # Apply projection bottleneck if configured
            if self.speaker_embedding_projection is not None:
                speaker_embedding = self.speaker_embedding_projection(speaker_embedding)

        # Initial projection
        x = self.initial_conv(z)
        x = self.initial_norm(x)
        x = self.initial_act(x)

        # Early FiLM conditioning (before attention/upsampling)
        # This allows speaker characteristics to influence the bottleneck representation
        if speaker_embedding is not None and self.speaker_embedding_dim > 0 and self.use_early_film:
            if hasattr(self, 'early_film_projection'):
                early_film_params = self.early_film_projection(speaker_embedding)
                out_c = x.shape[1]
                early_scale = early_film_params[:, :out_c].unsqueeze(-1).unsqueeze(-1)
                early_shift = early_film_params[:, out_c:].unsqueeze(-1).unsqueeze(-1)

                # Store raw (pre-bound) statistics for monitoring unbounded behavior
                if return_film_stats:
                    film_stats["early_scale_raw_mean"] = early_scale.mean().item()
                    film_stats["early_scale_raw_std"] = early_scale.std().item()
                    film_stats["early_scale_raw_min"] = early_scale.min().item()
                    film_stats["early_scale_raw_max"] = early_scale.max().item()
                    film_stats["early_shift_raw_mean"] = early_shift.mean().item()
                    film_stats["early_shift_raw_std"] = early_shift.std().item()
                    film_stats["early_shift_raw_min"] = early_shift.min().item()
                    film_stats["early_shift_raw_max"] = early_shift.max().item()
                    # Per-speaker variance: std across batch dim, averaged over channels
                    # If near 0, FiLM produces same output for all speaker embeddings
                    if early_scale.shape[0] > 1:  # Need multiple samples for variance
                        film_stats["early_scale_raw_speaker_var"] = early_scale.squeeze(-1).squeeze(-1).std(dim=0).mean().item()
                        film_stats["early_shift_raw_speaker_var"] = early_shift.squeeze(-1).squeeze(-1).std(dim=0).mean().item()

                if self.film_scale_bound > 0:
                    early_scale = self.film_scale_bound * torch.tanh(early_scale)
                if self.film_shift_bound > 0:
                    early_shift = self.film_shift_bound * torch.tanh(early_shift)

                # Store post-bound statistics
                if return_film_stats:
                    film_stats["early_scale_mean"] = early_scale.mean().item()
                    film_stats["early_scale_std"] = early_scale.std().item()
                    film_stats["early_scale_min"] = early_scale.min().item()
                    film_stats["early_scale_max"] = early_scale.max().item()
                    film_stats["early_shift_mean"] = early_shift.mean().item()
                    film_stats["early_shift_std"] = early_shift.std().item()
                    film_stats["early_shift_min"] = early_shift.min().item()
                    film_stats["early_shift_max"] = early_shift.max().item()

                x = x * (1 + early_scale) + early_shift

        # Bottleneck attention (with optional masking)
        attn_weights = None
        if self.use_attention:
            attn_result = self.attention(
                x,
                lengths=lengths,
                return_attention_weights=return_attention_weights,
            )
            if return_attention_weights:
                x, attn_weights = attn_result
            else:
                x = attn_result

        # Process through decoder stages
        for i, stage in enumerate(self.stages):
            for layer in stage:
                x = layer(x)

            # Apply FiLM conditioning after each stage
            if speaker_embedding is not None and self.speaker_embedding_dim > 0:
                film_params = self.speaker_projections[i](speaker_embedding)
                out_c = x.shape[1]
                scale = film_params[:, :out_c].unsqueeze(-1).unsqueeze(-1)
                shift = film_params[:, out_c:].unsqueeze(-1).unsqueeze(-1)

                # Store raw (pre-bound) statistics for monitoring unbounded behavior
                if return_film_stats:
                    film_stats[f"stage{i}_scale_raw_mean"] = scale.mean().item()
                    film_stats[f"stage{i}_scale_raw_std"] = scale.std().item()
                    film_stats[f"stage{i}_scale_raw_min"] = scale.min().item()
                    film_stats[f"stage{i}_scale_raw_max"] = scale.max().item()
                    film_stats[f"stage{i}_shift_raw_mean"] = shift.mean().item()
                    film_stats[f"stage{i}_shift_raw_std"] = shift.std().item()
                    film_stats[f"stage{i}_shift_raw_min"] = shift.min().item()
                    film_stats[f"stage{i}_shift_raw_max"] = shift.max().item()
                    # Per-speaker variance: std across batch dim, averaged over channels
                    # If near 0, FiLM produces same output for all speaker embeddings
                    if scale.shape[0] > 1:  # Need multiple samples for variance
                        film_stats[f"stage{i}_scale_raw_speaker_var"] = scale.squeeze(-1).squeeze(-1).std(dim=0).mean().item()
                        film_stats[f"stage{i}_shift_raw_speaker_var"] = shift.squeeze(-1).squeeze(-1).std(dim=0).mean().item()

                if self.film_scale_bound > 0:
                    scale = self.film_scale_bound * torch.tanh(scale)
                if self.film_shift_bound > 0:
                    shift = self.film_shift_bound * torch.tanh(shift)

                # Store post-bound statistics
                if return_film_stats:
                    film_stats[f"stage{i}_scale_mean"] = scale.mean().item()
                    film_stats[f"stage{i}_scale_std"] = scale.std().item()
                    film_stats[f"stage{i}_scale_min"] = scale.min().item()
                    film_stats[f"stage{i}_scale_max"] = scale.max().item()
                    film_stats[f"stage{i}_shift_mean"] = shift.mean().item()
                    film_stats[f"stage{i}_shift_std"] = shift.std().item()
                    film_stats[f"stage{i}_shift_min"] = shift.min().item()
                    film_stats[f"stage{i}_shift_max"] = shift.max().item()

                x = x * (1 + scale) + shift

        # Final output
        recon_x = self.final_conv(x)

        # Build return value based on what's requested
        if return_attention_weights and return_film_stats:
            return recon_x, attn_weights, film_stats
        elif return_attention_weights:
            return recon_x, attn_weights
        elif return_film_stats:
            return recon_x, film_stats

        return recon_x


# =============================================================================
# GuBERT Feature VAE Components (1D)
# =============================================================================

class Snake1d(nn.Module):
    """
    Snake activation for 1D inputs: x + (1/alpha) * sin^2(alpha * x)

    The alpha parameter is learnable per channel, allowing the network
    to adapt the periodicity to different frequency ranges.
    """
    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1), alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, T]
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2


class ResidualBlock1d(nn.Module):
    """
    Residual block for 1D sequential data (GuBERT features).

    Uses two conv layers with GroupNorm and a skip connection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel_size: int = 5,
        activation_fn: str = "silu",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = kernel_size // 2

        # Get activation
        if activation_fn == "snake":
            self.act1 = Snake1d(out_channels)
            self.act2 = Snake1d(out_channels)
        else:
            activation_type = get_activation_type(activation_fn)
            if activation_type in [activations.SwiGLU, activations.Snake]:
                self.act1 = activation_type(out_channels)
                self.act2 = activation_type(out_channels)
            else:
                self.act1 = activation_type()
                self.act2 = activation_type()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(max(1, out_channels // 4), out_channels)
        self.norm2 = nn.GroupNorm(max(1, out_channels // 4), out_channels)

        # Skip connection projection if channels change
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for module in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize second conv with smaller weights for stable residual learning
        self.conv2.weight.data *= 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip_proj is None else self.skip_proj(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.act2(x)

        return x


class GuBERTFeatureVAEEncoder(nn.Module):
    """
    VAE encoder for GuBERT features.

    Takes GuBERT features [B, T, D] and compresses to latent space.
    Uses Conv1D for temporal downsampling.

    Architecture:
    - Input projection: D -> intermediate_channels[0]
    - Downsampling stages with strided Conv1D
    - Optional residual blocks per stage
    - Output: mu, logvar [B, latent_dim, T']
    """
    def __init__(
        self,
        input_dim: int = 256,  # GuBERT feature dimension
        latent_dim: int = 32,  # Latent dimension per timestep
        intermediate_channels: list = None,
        kernel_sizes: list = None,
        strides: list = None,
        n_residual_blocks: int = 1,
        activation_fn: str = "silu",
        dropout: float = 0.1,
        logvar_clamp_max: float = 4.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.logvar_clamp_max = logvar_clamp_max

        # Defaults
        if intermediate_channels is None:
            intermediate_channels = [256, 256, 128]
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3]
        if strides is None:
            strides = [2, 2, 2]  # 8x total downsampling

        self.strides = strides
        channels = [input_dim] + intermediate_channels

        # Get activation type
        if activation_fn == "snake":
            self.get_activation = lambda c: Snake1d(c)
        else:
            activation_type = get_activation_type(activation_fn)
            if activation_type in [activations.SwiGLU, activations.Snake]:
                self.get_activation = lambda c: activation_type(c)
            else:
                self.get_activation = lambda c: activation_type()

        # Build encoder stages
        self.stages = nn.ModuleList()
        for i, (in_c, out_c, kernel_size, stride) in enumerate(
            zip(channels[:-1], channels[1:], kernel_sizes, strides)
        ):
            stage = nn.ModuleList()

            # Strided conv for downsampling
            padding = kernel_size // 2
            stage.append(nn.Conv1d(in_c, out_c, kernel_size, stride=stride, padding=padding))
            stage.append(nn.GroupNorm(max(1, out_c // 4), out_c))
            stage.append(self.get_activation(out_c))

            # Residual blocks
            for _ in range(n_residual_blocks):
                stage.append(ResidualBlock1d(out_c, out_c, kernel_size=kernel_size, activation_fn=activation_fn))

            if dropout > 0:
                stage.append(nn.Dropout(dropout))

            self.stages.append(stage)

        # Output projections for mu and logvar
        final_channels = intermediate_channels[-1]
        self.fc_mu = nn.Conv1d(final_channels, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = nn.Conv1d(final_channels, latent_dim, kernel_size=3, padding=1)

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

    def get_output_length(self, input_length: int) -> int:
        """Compute output temporal length given input length."""
        length = input_length
        for stride in self.strides:
            length = (length + stride - 1) // stride  # Ceiling division
        return length

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> tuple:
        """
        Args:
            x: [B, T, D] GuBERT features
            lengths: [B] optional valid lengths

        Returns:
            mu: [B, latent_dim, T'] latent mean
            logvar: [B, latent_dim, T'] latent log variance
            output_lengths: [B] downsampled lengths (if lengths provided)
        """
        # Permute to [B, D, T] for Conv1d
        x = x.permute(0, 2, 1)

        # Process through encoder stages
        for stage in self.stages:
            for layer in stage:
                x = layer(x)

        # Get mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10.0, max=self.logvar_clamp_max)

        # Compute output lengths if provided
        output_lengths = None
        if lengths is not None:
            output_lengths = lengths.clone()
            for stride in self.strides:
                output_lengths = (output_lengths + stride - 1) // stride

        return mu, logvar, output_lengths


class GuBERTFeatureVAEDecoder(nn.Module):
    """
    VAE decoder for GuBERT features.

    Takes latent [B, latent_dim, T'] and reconstructs features [B, T, D].
    Uses Upsample + Conv1D for temporal upsampling.

    Architecture:
    - Input projection: latent_dim -> intermediate_channels[0]
    - Upsampling stages with Upsample + Conv1D
    - Optional residual blocks per stage
    - Output projection: intermediate_channels[-1] -> output_dim
    """
    def __init__(
        self,
        latent_dim: int = 32,
        output_dim: int = 256,  # GuBERT feature dimension
        intermediate_channels: list = None,
        kernel_sizes: list = None,
        scale_factors: list = None,
        n_residual_blocks: int = 1,
        activation_fn: str = "silu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Defaults (mirror encoder)
        if intermediate_channels is None:
            intermediate_channels = [128, 256, 256]
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 5]
        if scale_factors is None:
            scale_factors = [2, 2, 2]  # 8x total upsampling

        self.scale_factors = scale_factors

        # Get activation type
        if activation_fn == "snake":
            self.get_activation = lambda c: Snake1d(c)
        else:
            activation_type = get_activation_type(activation_fn)
            if activation_type in [activations.SwiGLU, activations.Snake]:
                self.get_activation = lambda c: activation_type(c)
            else:
                self.get_activation = lambda c: activation_type()

        # Initial projection from latent
        self.initial_conv = nn.Conv1d(latent_dim, intermediate_channels[0], kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(max(1, intermediate_channels[0] // 4), intermediate_channels[0])
        self.initial_act = self.get_activation(intermediate_channels[0])

        # Build decoder stages
        self.stages = nn.ModuleList()
        all_channels = [intermediate_channels[0]] + intermediate_channels

        for i, (in_c, out_c, kernel_size, scale_factor) in enumerate(
            zip(all_channels[:-1], intermediate_channels, kernel_sizes, scale_factors)
        ):
            stage = nn.ModuleList()

            # Upsample
            stage.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))

            # Conv after upsample
            padding = kernel_size // 2
            stage.append(nn.Conv1d(in_c, out_c, kernel_size, padding=padding))
            stage.append(nn.GroupNorm(max(1, out_c // 4), out_c))
            stage.append(self.get_activation(out_c))

            # Residual blocks
            for _ in range(n_residual_blocks):
                stage.append(ResidualBlock1d(out_c, out_c, kernel_size=kernel_size, activation_fn=activation_fn))

            if dropout > 0:
                stage.append(nn.Dropout(dropout))

            self.stages.append(stage)

        # Final output projection
        self.final_conv = nn.Conv1d(intermediate_channels[-1], output_dim, kernel_size=3, padding=1)

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

    def get_output_length(self, input_length: int) -> int:
        """Compute output temporal length given latent length."""
        length = input_length
        for scale_factor in self.scale_factors:
            length = length * scale_factor
        return length

    def forward(self, z: torch.Tensor, target_length: int = None) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim, T'] latent tensor
            target_length: Optional target output length for trimming

        Returns:
            recon: [B, T, D] reconstructed GuBERT features
        """
        # Initial projection
        x = self.initial_conv(z)
        x = self.initial_norm(x)
        x = self.initial_act(x)

        # Process through decoder stages
        for stage in self.stages:
            for layer in stage:
                x = layer(x)

        # Final output projection
        x = self.final_conv(x)

        # Permute back to [B, T, D]
        x = x.permute(0, 2, 1)

        # Trim to target length if specified
        if target_length is not None and x.shape[1] > target_length:
            x = x[:, :target_length, :]

        return x


class GuBERTFeatureVAE(nn.Module):
    """
    Complete VAE for GuBERT features.

    Compresses speaker-invariant speech features to a lower-dimensional
    latent space while preserving linguistic/prosodic structure.

    Input: [B, T, D] GuBERT features
    Latent: [B, latent_dim, T'] compressed representation
    Output: [B, T, D] reconstructed features
    """
    def __init__(
        self,
        encoder: GuBERTFeatureVAEEncoder,
        decoder: GuBERTFeatureVAEDecoder,
        kl_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

    def encode(self, x: torch.Tensor, lengths: torch.Tensor = None) -> tuple:
        """Encode features to latent distribution."""
        return self.encoder(x, lengths)

    def decode(self, z: torch.Tensor, target_length: int = None) -> torch.Tensor:
        """Decode latent to features."""
        return self.decoder(z, target_length)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor = None,
        return_latent: bool = False,
    ) -> dict:
        """
        Forward pass through VAE.

        Args:
            x: [B, T, D] GuBERT features
            lengths: [B] optional valid lengths
            return_latent: If True, also return latent variables

        Returns:
            dict with:
                - recon: [B, T, D] reconstructed features
                - mu: [B, latent_dim, T'] latent mean
                - logvar: [B, latent_dim, T'] latent log variance
                - z: [B, latent_dim, T'] sampled latent (if return_latent)
                - output_lengths: [B] downsampled lengths (if lengths provided)
        """
        target_length = x.shape[1]

        # Encode
        mu, logvar, output_lengths = self.encode(x, lengths)

        # Sample
        z = self.reparameterize(mu, logvar)

        # Decode
        recon = self.decode(z, target_length)

        result = {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "output_lengths": output_lengths,
        }

        if return_latent:
            result["z"] = z

        return result

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> dict:
        """
        Compute VAE loss.

        Args:
            x: [B, T, D] original features
            recon: [B, T, D] reconstructed features
            mu: [B, latent_dim, T'] latent mean
            logvar: [B, latent_dim, T'] latent log variance
            lengths: [B] optional valid lengths for masked loss

        Returns:
            dict with reconstruction_loss, kl_loss, total_loss
        """
        # Reconstruction loss (MSE)
        if lengths is not None:
            # Masked reconstruction loss
            mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(x)
            recon_loss = F.mse_loss(recon * mask, x * mask, reduction='sum') / mask.sum()
        else:
            recon_loss = F.mse_loss(recon, x)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Model Config Lookups
# =============================================================================

model_config_lookup = {
    # Tiny config for testing - minimal channels, no attention, fast forward pass
    "tiny_test": lambda latent_channels, speaker_embedding_dim=0, speaker_embedding_proj_dim=0, normalize_speaker_embedding=False, film_scale_bound=0.0, film_shift_bound=0.0, zero_init_film_bias=True, film_no_bias=False, learn_speaker_embedding=False, learned_speaker_dim=256, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[16, 32, 64],
            kernel_sizes=[(3, 5), (3, 5), (3, 3)],
            strides=[(2, 4), (2, 3), (2, 1)],
            n_residual_blocks=0,
            use_attention=False,
            activation_fn="gelu",
            learn_speaker_embedding=learn_speaker_embedding,
            learned_speaker_dim=learned_speaker_dim,
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[48, 24, 12],
            scale_factors=[(2, 1), (2, 3), (2, 4)],
            kernel_sizes=[(3, 3), (3, 5), (3, 7)],
            n_residual_blocks=0,
            use_attention=False,
            activation_fn="gelu",
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
        ),
        **kwargs
    ),
    "medium_no_attn": lambda latent_channels, speaker_embedding_dim=0, speaker_embedding_proj_dim=0, normalize_speaker_embedding=False, film_scale_bound=0.0, film_shift_bound=0.0, zero_init_film_bias=True, film_no_bias=False, learn_speaker_embedding=False, learned_speaker_dim=256, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[168, 336, 672],
            kernel_sizes=[(3, 7), (3, 7), (3, 5)],
            strides=[(2, 4), (2, 3), (2, 1)],  # 4*3*1 = 12x time compression (1875 -> ~156)
            n_residual_blocks=0,
            use_attention=False,
            activation_fn="snake",
            learn_speaker_embedding=learn_speaker_embedding,
            learned_speaker_dim=learned_speaker_dim,
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[384, 192, 96],
            scale_factors=[(2, 1), (2, 3), (2, 4)],  # 1*3*4 = 12x time upsampling
            kernel_sizes=[(3, 5), (3, 7), (3, 9)],  # increased last kernel for stride-4 upsample
            n_residual_blocks=0,
            use_attention=False,
            activation_fn="snake",
            # When learn_speaker_embedding=True, decoder uses learned_speaker_dim for FiLM
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
        ),
        **kwargs
    ),
    # Creates a ~13M parameter VAE with ~4M encoder and ~9M decoder - includes snake activations, bottleneck attention, and FiLM speaker embedding conditioning (if speaker_embedding_dim > 0)
    # but no residual blocks
    "large": lambda latent_channels, speaker_embedding_dim=0, speaker_embedding_proj_dim=0, normalize_speaker_embedding=False, film_scale_bound=0.0, film_shift_bound=0.0, zero_init_film_bias=True, film_no_bias=False, learn_speaker_embedding=False, learned_speaker_dim=256, **kwargs: VAE(
        encoder=AudioVAEEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            intermediate_channels=[128, 256, 512],
            kernel_sizes=[(3, 7), (3, 7), (3, 5)],
            strides=[(2, 4), (2, 3), (2, 1)],  # 4*3*1 = 12x time compression (1875 -> ~156)
            n_residual_blocks=0,
            use_attention=True,
            attention_heads=6,
            activation_fn="snake",
            learn_speaker_embedding=learn_speaker_embedding,
            learned_speaker_dim=learned_speaker_dim,
        ),
        decoder=AudioVAEDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            intermediate_channels=[352, 176, 88],
            scale_factors=[(2, 1), (2, 3), (2, 4)],  # 1*3*4 = 12x time upsampling
            kernel_sizes=[(3, 5), (3, 7), (3, 9)],  # increased last kernel for stride-4 upsample
            n_residual_blocks=0,
            use_attention=True,
            attention_heads=6,
            activation_fn="snake",
            # When learn_speaker_embedding=True, decoder uses learned_speaker_dim for FiLM
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
        ),
        **kwargs
    ),
}


# GuBERT Feature VAE configs
# These operate on GuBERT features [B, T, D] instead of spectrograms
gubert_feature_vae_config_lookup = {
    # Tiny config for testing
    "tiny": lambda input_dim=128, latent_dim=16, **kwargs: GuBERTFeatureVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[64, 64],
            kernel_sizes=[3, 3],
            strides=[2, 2],  # 4x downsampling
            n_residual_blocks=0,
            activation_fn="silu",
            dropout=0.0,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            intermediate_channels=[64, 64],
            kernel_sizes=[3, 3],
            scale_factors=[2, 2],  # 4x upsampling
            n_residual_blocks=0,
            activation_fn="silu",
            dropout=0.0,
        ),
        **kwargs
    ),

    # Small config - good balance of capacity and efficiency
    "small": lambda input_dim=256, latent_dim=32, **kwargs: GuBERTFeatureVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[256, 192, 128],
            kernel_sizes=[5, 5, 3],
            strides=[2, 2, 2],  # 8x downsampling
            n_residual_blocks=1,
            activation_fn="silu",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            intermediate_channels=[128, 192, 256],
            kernel_sizes=[3, 5, 5],
            scale_factors=[2, 2, 2],  # 8x upsampling
            n_residual_blocks=1,
            activation_fn="silu",
            dropout=0.1,
        ),
        **kwargs
    ),

    # Medium config - more capacity
    "medium": lambda input_dim=256, latent_dim=48, **kwargs: GuBERTFeatureVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[320, 256, 192],
            kernel_sizes=[7, 5, 3],
            strides=[2, 2, 2],  # 8x downsampling
            n_residual_blocks=2,
            activation_fn="snake",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            intermediate_channels=[192, 256, 320],
            kernel_sizes=[3, 5, 7],
            scale_factors=[2, 2, 2],  # 8x upsampling
            n_residual_blocks=2,
            activation_fn="snake",
            dropout=0.1,
        ),
        **kwargs
    ),

    # Large config - high capacity for larger GuBERT models
    "large": lambda input_dim=512, latent_dim=64, **kwargs: GuBERTFeatureVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[512, 384, 256],
            kernel_sizes=[7, 5, 3],
            strides=[2, 2, 2],  # 8x downsampling
            n_residual_blocks=2,
            activation_fn="snake",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            intermediate_channels=[256, 384, 512],
            kernel_sizes=[3, 5, 7],
            scale_factors=[2, 2, 2],  # 8x upsampling
            n_residual_blocks=2,
            activation_fn="snake",
            dropout=0.1,
        ),
        **kwargs
    ),

    # XL config - for very large GuBERT models with more compression
    "xlarge": lambda input_dim=512, latent_dim=64, **kwargs: GuBERTFeatureVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[512, 384, 256, 192],
            kernel_sizes=[7, 5, 5, 3],
            strides=[2, 2, 2, 2],  # 16x downsampling
            n_residual_blocks=2,
            activation_fn="snake",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            intermediate_channels=[192, 256, 384, 512],
            kernel_sizes=[3, 5, 5, 7],
            scale_factors=[2, 2, 2, 2],  # 16x upsampling
            n_residual_blocks=2,
            activation_fn="snake",
            dropout=0.1,
        ),
        **kwargs
    ),
}


def create_gubert_feature_vae(
    config: str = "small",
    input_dim: int = None,
    latent_dim: int = None,
    **kwargs,
) -> GuBERTFeatureVAE:
    """
    Create a GuBERT Feature VAE from a config name.

    Args:
        config: Config name (tiny, small, medium, large, xlarge)
        input_dim: Override GuBERT feature dimension
        latent_dim: Override latent dimension
        **kwargs: Additional arguments (e.g., kl_weight)

    Returns:
        GuBERTFeatureVAE instance
    """
    if config not in gubert_feature_vae_config_lookup:
        raise ValueError(f"Unknown config: {config}. Available: {list(gubert_feature_vae_config_lookup.keys())}")

    factory_kwargs = {}
    if input_dim is not None:
        factory_kwargs["input_dim"] = input_dim
    if latent_dim is not None:
        factory_kwargs["latent_dim"] = latent_dim
    factory_kwargs.update(kwargs)

    return gubert_feature_vae_config_lookup[config](**factory_kwargs)
