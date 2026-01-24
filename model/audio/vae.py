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


def masked_instance_norm(
    x: torch.Tensor,
    lengths: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Instance normalization that only computes statistics over valid (non-padded) frames.

    For audio VAE, this ensures that normalization statistics are not corrupted by
    padding silence, which can cause energy artifacts at utterance boundaries.

    Args:
        x: [B, C, T] input tensor (mel spectrogram with C mel bins and T time frames)
        lengths: [B] tensor of valid lengths per sample. If None, uses standard instance norm.
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as input
    """
    if lengths is None:
        # Fall back to standard instance norm if no lengths provided
        return F.instance_norm(x, eps=eps)

    B, C, T = x.shape
    device = x.device

    # Create mask: [B, 1, T] where 1 = valid, 0 = padding
    time_indices = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
    mask = (time_indices < lengths.unsqueeze(1)).unsqueeze(1).float()  # [B, 1, T]

    # Compute valid counts per sample: [B, 1, 1]
    valid_counts = mask.sum(dim=2, keepdim=True).clamp(min=1)  # [B, 1, 1]

    # Masked mean: sum(x * mask) / valid_count, per channel per sample
    # x * mask zeros out padding, then sum over T and divide by valid count
    masked_sum = (x * mask).sum(dim=2, keepdim=True)  # [B, C, 1]
    mean = masked_sum / valid_counts  # [B, C, 1]

    # Masked variance: sum((x - mean)^2 * mask) / valid_count
    diff = x - mean  # [B, C, T]
    masked_sq_sum = ((diff ** 2) * mask).sum(dim=2, keepdim=True)  # [B, C, 1]
    var = masked_sq_sum / valid_counts  # [B, C, 1]

    # Normalize
    x_norm = (x - mean) / (var + eps).sqrt()

    # Zero out padding positions (optional but cleaner)
    x_norm = x_norm * mask

    return x_norm


class AudioVAEEncoder(nn.Module):
    """
    Audio VAE encoder with:
    - Residual blocks for better gradient flow
    - Snake activation for audio-specific periodicity
    - Optional bottleneck attention for long-range dependencies
    - Larger receptive fields via configurable kernel sizes
    - Optional instance normalization for speaker-invariant features

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
        # Instance normalization for input mel spectrogram (speaker-invariant features)
        use_instance_norm: bool = False,  # Normalize input mel to strip per-utterance speaker statistics (like CMVN)
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
        self.use_instance_norm = use_instance_norm

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
        # Apply instance normalization to input mel spectrogram (if enabled)
        # Treats mel bins as channels and normalizes each bin across time independently
        # This strips per-utterance temporal statistics (similar to CMVN)
        # Uses masked normalization when lengths provided to avoid contaminating
        # statistics with padded silence (which can cause energy artifacts at boundaries)
        if self.use_instance_norm:
            # x: [B, 1, M, T] -> [B, M, T] -> instance_norm -> [B, 1, M, T]
            x_squeezed = x.squeeze(1)  # [B, M, T]
            x_squeezed = masked_instance_norm(x_squeezed, lengths=lengths)
            x = x_squeezed.unsqueeze(1)  # [B, 1, M, T]

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


class F0Predictor(nn.Module):
    """
    Predicts F0 (fundamental frequency) contour from speaker embedding + GuBERT features.

    Speaker embedding provides F0 range/characteristics (who is speaking).
    GuBERT features provide prosodic timing cues (where stress/emphasis occurs).

    Output is per-frame: log F0 values and voiced/unvoiced probability.
    """
    def __init__(
        self,
        speaker_dim: int = 192,
        gubert_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 3,
        kernel_size: int = 5,
    ):
        super().__init__()

        self.speaker_dim = speaker_dim
        self.gubert_dim = gubert_dim
        self.hidden_dim = hidden_dim

        # Project speaker embedding to F0-relevant representation
        self.speaker_proj = nn.Linear(speaker_dim, hidden_dim)

        # Project GuBERT features (prosodic timing info)
        self.gubert_proj = nn.Conv1d(gubert_dim, hidden_dim, kernel_size=1)

        # Combine and predict F0 contour with residual convolutions
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
            )
            for _ in range(n_layers)
        ])

        # Output: log_f0 (continuous) + voiced_logit
        self.output_proj = nn.Conv1d(hidden_dim, 2, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # Initialize output layer to produce reasonable F0 values
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            # Initialize log_f0 bias to ~5.0 (≈150 Hz), voiced bias to 0
            self.output_proj.bias.data[0] = 5.0  # log(150) ≈ 5.0
            self.output_proj.bias.data[1] = 0.0

    def forward(
        self,
        speaker_embedding: torch.Tensor,
        gubert_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            speaker_embedding: [B, speaker_dim] or [B, 1, speaker_dim]
            gubert_features: [B, gubert_dim, T']

        Returns:
            log_f0: [B, T'] - log fundamental frequency (in log Hz)
            voiced_prob: [B, T'] - probability frame is voiced (0-1)
        """
        if speaker_embedding.dim() == 3:
            speaker_embedding = speaker_embedding.squeeze(1)

        T = gubert_features.size(-1)

        # Speaker embedding -> F0 characteristics, broadcast across time
        spk = self.speaker_proj(speaker_embedding)  # [B, hidden]
        spk = spk.unsqueeze(-1).expand(-1, -1, T)   # [B, hidden, T']

        # GuBERT -> prosodic timing patterns
        content = self.gubert_proj(gubert_features)  # [B, hidden, T']

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
    def __init__(
        self,
        embedding_dim: int = 64,
        n_harmonics: int = 6,
        sample_rate: int = 16000,
        hop_length: int = 256,
        use_harmonic_sinusoids: bool = False,  # Set True for full harmonic sin/cos embedding
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.use_harmonic_sinusoids = use_harmonic_sinusoids

        # Time step in seconds for each frame
        self.frame_duration = hop_length / sample_rate

        if use_harmonic_sinusoids:
            # Harmonic sinusoids: sin/cos for each harmonic + log_f0 + voiced = 2*n_harmonics + 2
            input_dim = 2 * n_harmonics + 2
        else:
            # Simplified: log_f0_norm, log_f0_norm^2, voiced, voiced^2
            input_dim = 4

        self.proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Learnable output scale - starts small to avoid dominating the latent
        self.output_scale = nn.Parameter(torch.tensor(0.1))

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

        if self.use_harmonic_sinusoids:
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
            for h in range(1, self.n_harmonics + 1):
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

        else:
            # Simplified version (no sinusoids)
            log_f0_norm = (log_f0 - 5.0) / 1.5

            features = torch.stack([
                log_f0_norm,
                log_f0_norm ** 2,
                voiced_prob,
                voiced_prob ** 2,
            ], dim=-1)  # [B, T, 4]

            voicing_gate = voiced_prob.unsqueeze(-1) ** 2
            features = features * voicing_gate

        emb = self.proj(features)  # [B, T, embedding_dim]

        # Scale output
        emb = emb * self.output_scale

        return emb.transpose(1, 2)  # [B, embedding_dim, T]


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
        output_dim: int = 288,  # GuBERT feature dimension
        intermediate_channels: list = None,
        kernel_sizes: list = None,
        scale_factors: list = None,
        n_residual_blocks: int = 1,
        activation_fn: str = "silu",
        dropout: float = 0.1,
        speaker_embedding_dim: int = 192,
        speaker_embedding_proj_dim: int = 0,  # If > 0, project speaker embedding to this dim before FiLM
        normalize_speaker_embedding: bool = True,
        film_scale_bound: float = 0.5,
        film_shift_bound: float = 0.5,
        zero_init_film_bias: bool = True,
        film_no_bias: bool = False,
        # F0 conditioning (Option 1: concatenate to latent input)
        f0_conditioning_dim: int = 0,  # If > 0, expect F0 embedding to be concatenated to latent
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.f0_conditioning_dim = f0_conditioning_dim
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_embedding_proj_dim = speaker_embedding_proj_dim
        self.normalize_speaker_embedding = normalize_speaker_embedding
        self.film_scale_bound = film_scale_bound
        self.film_shift_bound = film_shift_bound
        self.zero_init_film_bias = zero_init_film_bias
        self.film_no_bias = film_no_bias

        # Determine the effective speaker embedding dimension for FiLM layers
        # If proj_dim > 0, we project down before FiLM; otherwise use full dim
        self.film_speaker_dim = speaker_embedding_proj_dim if speaker_embedding_proj_dim > 0 else speaker_embedding_dim

        # Input dimension to decoder: latent + optional F0 conditioning
        decoder_input_dim = latent_dim + f0_conditioning_dim

        # Speaker embedding projection (bottleneck to reduce params in FiLM layers)
        self.speaker_embedding_projection = None
        if speaker_embedding_dim > 0 and speaker_embedding_proj_dim > 0 and speaker_embedding_proj_dim != speaker_embedding_dim:
            self.speaker_embedding_projection = nn.Sequential(
                nn.Linear(speaker_embedding_dim, speaker_embedding_proj_dim),
                nn.SiLU(),
            )

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

        # Initial projection from latent (+ optional F0 conditioning)
        self.initial_conv = nn.Conv1d(decoder_input_dim, intermediate_channels[0], kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(max(1, intermediate_channels[0] // 4), intermediate_channels[0])
        self.initial_act = self.get_activation(intermediate_channels[0])

        # Early FiLM projection (applied before attention/upsampling)
        # This allows speaker info to influence the bottleneck representation
        # Uses film_speaker_dim (projected dim if projection enabled, else full dim)
        if speaker_embedding_dim > 0:
            self.early_film_projection = nn.Sequential(
                nn.Linear(self.film_speaker_dim, self.film_speaker_dim, bias=not film_no_bias),
                nn.SiLU(),
                nn.Linear(self.film_speaker_dim, intermediate_channels[0] * 2, bias=not film_no_bias),
            )

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

            if self.film_speaker_dim > 0:
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

    def get_output_length(self, input_length: int) -> int:
        """Compute output temporal length given latent length."""
        length = input_length
        for scale_factor in self.scale_factors:
            length = length * scale_factor
        return length

    def forward(
        self,
        z: torch.Tensor,
        speaker_embedding=None,
        f0_embedding: torch.Tensor = None,
        return_film_stats: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim, T'] latent tensor
            speaker_embedding: [B, speaker_dim] or [B, 1, speaker_dim] speaker embedding for FiLM
            f0_embedding: [B, f0_conditioning_dim, T'] optional F0 conditioning (concatenated to z)

        Returns:
            recon: [B, output_dim, T] reconstructed features (e.g., mel spectrogram)
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

        # Concatenate F0 conditioning to latent if provided
        if f0_embedding is not None and self.f0_conditioning_dim > 0:
            # Ensure F0 embedding matches latent temporal dimension
            if f0_embedding.size(-1) != z.size(-1):
                f0_embedding = F.interpolate(
                    f0_embedding, size=z.size(-1), mode='linear', align_corners=False
                )
            decoder_input = torch.cat([z, f0_embedding], dim=1)  # [B, latent_dim + f0_dim, T']
        else:
            decoder_input = z

        # Initial projection
        x = self.initial_conv(decoder_input)
        x = self.initial_norm(x)
        x = self.initial_act(x)

        # Early FiLM conditioning (before attention/upsampling)
        # This allows speaker characteristics to influence the bottleneck representation
        if speaker_embedding is not None and self.speaker_embedding_dim > 0:
            if hasattr(self, 'early_film_projection'):
                early_film_params = self.early_film_projection(speaker_embedding)
                out_c = x.shape[1]
                early_scale = early_film_params[:, :out_c].unsqueeze(-1)
                early_shift = early_film_params[:, out_c:].unsqueeze(-1)

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
                        film_stats["early_scale_raw_speaker_var"] = early_scale.squeeze(-1).std(dim=0).mean().item()
                        film_stats["early_shift_raw_speaker_var"] = early_shift.squeeze(-1).std(dim=0).mean().item()

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

        # Process through decoder stages
        for i, stage in enumerate(self.stages):
            # Process all layers in stage
            for layer in stage:
                x = layer(x)

            # Apply FiLM conditioning after each stage (not after each layer!)
            if speaker_embedding is not None and self.speaker_embedding_dim > 0:
                film_params = self.speaker_projections[i](speaker_embedding)
                out_c = x.shape[1]
                scale = film_params[:, :out_c].unsqueeze(-1)
                shift = film_params[:, out_c:].unsqueeze(-1)

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
                        film_stats[f"stage{i}_scale_raw_speaker_var"] = scale.squeeze(-1).std(dim=0).mean().item()
                        film_stats[f"stage{i}_shift_raw_speaker_var"] = shift.squeeze(-1).std(dim=0).mean().item()

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

        # Final output projection
        recon_x = self.final_conv(x)

        # Build return value based on what's requested
        if return_film_stats:
            return recon_x, film_stats

        return recon_x


class GuBERTFeatureVAEDecoder2D(nn.Module):
    """
    Hybrid VAE decoder that transitions from 1D to 2D convolutions.

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
    def __init__(
        self,
        latent_dim: int = 16,
        output_dim: int = 80,  # n_mels
        # 1D processing stage (before reshape to 2D)
        conv1d_channels: int = 320,  # Should be divisible by initial_freq_bins
        conv1d_n_residual_blocks: int = 2,
        conv1d_kernel_size: int = 5,
        conv1d_upsample_factor: int = 1,  # Optional time upsampling in 1D stage
        # 1D to 2D transition
        initial_freq_bins: int = 10,  # Frequency bins after reshape (conv1d_channels / channels_2d_initial)
        # 2D processing stages
        intermediate_channels_2d: list = None,  # e.g., [64, 128, 256]
        kernel_sizes_2d: list = None,  # e.g., [(3, 5), (3, 5), (3, 5)]
        scale_factors_2d: list = None,  # e.g., [(2, 2), (2, 2), (2, 1)] for freq and time
        n_residual_blocks_2d: int = 2,
        # General
        activation_fn: str = "silu",
        dropout: float = 0.1,
        # Speaker conditioning (FiLM)
        speaker_embedding_dim: int = 192,
        speaker_embedding_proj_dim: int = 0,
        normalize_speaker_embedding: bool = True,
        film_scale_bound: float = 0.5,
        film_shift_bound: float = 0.5,
        zero_init_film_bias: bool = True,
        film_no_bias: bool = False,
        # F0 conditioning
        f0_conditioning_dim: int = 0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.f0_conditioning_dim = f0_conditioning_dim
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_embedding_proj_dim = speaker_embedding_proj_dim
        self.normalize_speaker_embedding = normalize_speaker_embedding
        self.film_scale_bound = film_scale_bound
        self.film_shift_bound = film_shift_bound
        self.zero_init_film_bias = zero_init_film_bias
        self.film_no_bias = film_no_bias
        self.initial_freq_bins = initial_freq_bins
        self.conv1d_upsample_factor = conv1d_upsample_factor

        # Effective speaker dim for FiLM
        self.film_speaker_dim = speaker_embedding_proj_dim if speaker_embedding_proj_dim > 0 else speaker_embedding_dim

        # Input dim is just latent (F0 is injected after 1D→2D transition)
        decoder_input_dim = latent_dim

        # Defaults for 2D stages
        if intermediate_channels_2d is None:
            intermediate_channels_2d = [64, 128, 256]
        if kernel_sizes_2d is None:
            kernel_sizes_2d = [(3, 5), (3, 5), (3, 5)]
        if scale_factors_2d is None:
            # Default: 8x freq (10→80), 4x time (T'→4T')
            scale_factors_2d = [(2, 2), (2, 2), (2, 1)]

        self.scale_factors_2d = scale_factors_2d

        # Compute channels after 1D→2D reshape
        channels_2d_initial = conv1d_channels // initial_freq_bins
        self.channels_2d_initial = channels_2d_initial
        assert conv1d_channels % initial_freq_bins == 0, \
            f"conv1d_channels ({conv1d_channels}) must be divisible by initial_freq_bins ({initial_freq_bins})"

        # F0 conditioning projection: inject after 1D→2D transition
        # Projects F0 embedding to 2D spatial dimensions for additive conditioning
        self.f0_to_2d_projection = None
        if f0_conditioning_dim > 0:
            # Project [B, f0_dim, T'] → [B, channels_2d * freq_bins, T'] → reshape to [B, C, H, T']
            self.f0_to_2d_projection = nn.Sequential(
                nn.Conv1d(f0_conditioning_dim, channels_2d_initial * initial_freq_bins, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(channels_2d_initial * initial_freq_bins, channels_2d_initial * initial_freq_bins, kernel_size=3, padding=1),
            )

        # Speaker embedding projection
        self.speaker_embedding_projection = None
        if speaker_embedding_dim > 0 and speaker_embedding_proj_dim > 0 and speaker_embedding_proj_dim != speaker_embedding_dim:
            self.speaker_embedding_projection = nn.Sequential(
                nn.Linear(speaker_embedding_dim, speaker_embedding_proj_dim),
                nn.SiLU(),
            )

        # === 1D Processing Stage ===
        # Activation helper
        if activation_fn == "snake":
            self.get_activation_1d = lambda c: Snake1d(c)
            self.get_activation_2d = lambda c: Snake2d(c)
        else:
            activation_type = get_activation_type(activation_fn)
            if activation_type in [activations.SwiGLU, activations.Snake]:
                self.get_activation_1d = lambda c: activation_type(c)
                self.get_activation_2d = lambda c: activation_type(c)
            else:
                self.get_activation_1d = lambda _: activation_type()
                self.get_activation_2d = lambda _: activation_type()

        # Initial 1D projection
        self.conv1d_initial = nn.Conv1d(decoder_input_dim, conv1d_channels, kernel_size=3, padding=1)
        self.conv1d_initial_norm = nn.GroupNorm(max(1, conv1d_channels // 4), conv1d_channels)
        self.conv1d_initial_act = self.get_activation_1d(conv1d_channels)

        # Optional 1D upsampling
        self.conv1d_upsample = None
        if conv1d_upsample_factor > 1:
            self.conv1d_upsample = nn.Sequential(
                nn.Upsample(scale_factor=conv1d_upsample_factor, mode='nearest'),
                nn.Conv1d(conv1d_channels, conv1d_channels, kernel_size=conv1d_kernel_size, padding=conv1d_kernel_size // 2),
                nn.GroupNorm(max(1, conv1d_channels // 4), conv1d_channels),
                self.get_activation_1d(conv1d_channels),
            )

        # 1D residual blocks
        self.conv1d_residual_blocks = nn.ModuleList([
            ResidualBlock1d(conv1d_channels, conv1d_channels, kernel_size=conv1d_kernel_size, activation_fn=activation_fn)
            for _ in range(conv1d_n_residual_blocks)
        ])

        # Early FiLM (in 1D stage)
        if speaker_embedding_dim > 0:
            self.early_film_projection = nn.Sequential(
                nn.Linear(self.film_speaker_dim, self.film_speaker_dim, bias=not film_no_bias),
                nn.SiLU(),
                nn.Linear(self.film_speaker_dim, conv1d_channels * 2, bias=not film_no_bias),
            )

        # === 2D Processing Stages ===
        # FiLM projections for 2D stages
        if speaker_embedding_dim > 0:
            self.speaker_projections_2d = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.film_speaker_dim, self.film_speaker_dim, bias=not film_no_bias),
                    nn.SiLU(),
                    nn.Linear(self.film_speaker_dim, out_c * 2, bias=not film_no_bias),
                )
                for out_c in intermediate_channels_2d
            ])

        # Build 2D stages
        self.stages_2d = nn.ModuleList()
        all_channels_2d = [channels_2d_initial] + intermediate_channels_2d

        for in_c, out_c, kernel_size, scale_factor in zip(
            all_channels_2d[:-1], intermediate_channels_2d, kernel_sizes_2d, scale_factors_2d
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
            for _ in range(n_residual_blocks_2d):
                stage.append(ResidualBlock2d(out_c, out_c, kernel_size=kernel_size, activation_fn=activation_fn))

            if dropout > 0:
                stage.append(nn.Dropout2d(dropout))

            self.stages_2d.append(stage)

        # Final output conv
        self.final_conv = nn.Conv2d(intermediate_channels_2d[-1], 1, kernel_size=3, padding=1)

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
        if self.speaker_embedding_dim > 0:
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
                    if self.zero_init_film_bias:
                        nn.init.zeros_(linear.weight)
                    else:
                        nn.init.normal_(linear.weight, mean=0.0, std=0.02)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)

    def get_output_length(self, input_length: int) -> int:
        """Compute output temporal length given latent length."""
        length = input_length
        if self.conv1d_upsample_factor > 1:
            length = length * self.conv1d_upsample_factor
        for scale_factor in self.scale_factors_2d:
            length = length * scale_factor[1]  # Time dimension
        return length

    def forward(
        self,
        z: torch.Tensor,
        speaker_embedding=None,
        f0_embedding: torch.Tensor = None,
        return_film_stats: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim, T'] latent tensor (F0 is NOT concatenated here)
            speaker_embedding: [B, speaker_dim] or [B, 1, speaker_dim] speaker embedding for FiLM
            f0_embedding: [B, f0_conditioning_dim, T'] F0 conditioning (injected after 1D→2D transition)

        Returns:
            recon: [B, 1, n_mels, T] reconstructed mel spectrogram
        """
        film_stats = {} if return_film_stats else None

        # Process speaker embedding
        if speaker_embedding is not None:
            if speaker_embedding.dim() == 3:
                speaker_embedding = speaker_embedding.squeeze(1)
            if self.normalize_speaker_embedding:
                speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)
            if self.speaker_embedding_projection is not None:
                speaker_embedding = self.speaker_embedding_projection(speaker_embedding)

        # === 1D Processing ===
        x = self.conv1d_initial(z)
        x = self.conv1d_initial_norm(x)
        x = self.conv1d_initial_act(x)

        # Early FiLM (1D)
        if speaker_embedding is not None and self.speaker_embedding_dim > 0:
            film_params = self.early_film_projection(speaker_embedding)
            channels = x.shape[1]
            scale = film_params[:, :channels].unsqueeze(-1)
            shift = film_params[:, channels:].unsqueeze(-1)

            if return_film_stats:
                film_stats["early_scale_raw_mean"] = scale.mean().item()
                film_stats["early_shift_raw_mean"] = shift.mean().item()

            if self.film_scale_bound > 0:
                scale = self.film_scale_bound * torch.tanh(scale)
            if self.film_shift_bound > 0:
                shift = self.film_shift_bound * torch.tanh(shift)

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
        channels_2d = C // self.initial_freq_bins
        x = x.view(B, channels_2d, self.initial_freq_bins, T)

        # Inject F0 conditioning after reshape to 2D (additive)
        # F0 can now influence frequency-specific features
        if f0_embedding is not None and self.f0_to_2d_projection is not None:
            # Match temporal dimension if needed (after any 1D upsampling)
            if f0_embedding.shape[-1] != T:
                f0_embedding = F.interpolate(f0_embedding, size=T, mode='linear', align_corners=False)
            # Project F0 to 2D: [B, f0_dim, T] → [B, C*H, T] → [B, C, H, T]
            f0_2d = self.f0_to_2d_projection(f0_embedding)  # [B, C*H, T]
            f0_2d = f0_2d.view(B, self.channels_2d_initial, self.initial_freq_bins, T)
            x = x + f0_2d

        # === 2D Processing ===
        for i, stage in enumerate(self.stages_2d):
            for layer in stage:
                x = layer(x)

            # FiLM conditioning after each 2D stage
            if speaker_embedding is not None and self.speaker_embedding_dim > 0:
                film_params = self.speaker_projections_2d[i](speaker_embedding)
                out_c = x.shape[1]
                scale = film_params[:, :out_c].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                shift = film_params[:, out_c:].unsqueeze(-1).unsqueeze(-1)

                if return_film_stats:
                    film_stats[f"stage2d_{i}_scale_raw_mean"] = scale.mean().item()
                    film_stats[f"stage2d_{i}_shift_raw_mean"] = shift.mean().item()

                if self.film_scale_bound > 0:
                    scale = self.film_scale_bound * torch.tanh(scale)
                if self.film_shift_bound > 0:
                    shift = self.film_shift_bound * torch.tanh(shift)

                x = x * (1 + scale) + shift

        # Final output
        x = self.final_conv(x)  # [B, 1, n_mels, T]

        if return_film_stats:
            return x, film_stats

        return x


class GuBERTFeatureVAEDecoderMostly1D(nn.Module):
    """
    VAE decoder that is mostly 1D with a small 2D refinement at the end.

    Takes 1D latent [B, latent_dim, T'] and produces 2D mel spectrogram [B, 1, n_mels, T].

    Architecture:
    1. 1D processing: latent (+F0) → Conv1d upsampling stages with FiLM
    2. Reshape: [B, n_mels, T] → [B, 1, n_mels, T]
    3. 2D refinement: small Conv2d block for harmonic refinement (no upsampling)

    This architecture:
    - Preserves temporal dynamics throughout (better for content/phonemes)
    - Applies FiLM speaker conditioning early and throughout 1D stages
    - Uses 2D only at the end for frequency-local pattern refinement (harmonics)
    """
    def __init__(
        self,
        latent_dim: int = 16,
        output_dim: int = 80,  # n_mels
        # 1D processing stages (does the heavy lifting)
        intermediate_channels: list = None,  # e.g., [128, 256, 256, 128, 80]
        kernel_sizes: list = None,
        scale_factors: list = None,  # e.g., [2, 2, 1, 1] for 4x upsampling
        n_residual_blocks: int = 2,
        # 2D refinement at the end (small, no upsampling)
        conv2d_channels: list = None,  # e.g., [64, 32] - shallow refinement
        conv2d_kernel_size: tuple = (3, 5),
        conv2d_n_residual_blocks: int = 1,
        # General
        activation_fn: str = "silu",
        dropout: float = 0.1,
        # Speaker conditioning (FiLM) - applied throughout 1D stages
        speaker_embedding_dim: int = 192,
        speaker_embedding_proj_dim: int = 0,
        normalize_speaker_embedding: bool = True,
        film_scale_bound: float = 0.5,
        film_shift_bound: float = 0.5,
        zero_init_film_bias: bool = True,
        film_no_bias: bool = False,
        # F0 conditioning (concatenated to latent input)
        f0_conditioning_dim: int = 0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.f0_conditioning_dim = f0_conditioning_dim
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_embedding_proj_dim = speaker_embedding_proj_dim
        self.normalize_speaker_embedding = normalize_speaker_embedding
        self.film_scale_bound = film_scale_bound
        self.film_shift_bound = film_shift_bound
        self.zero_init_film_bias = zero_init_film_bias
        self.film_no_bias = film_no_bias

        # Effective speaker dim for FiLM
        self.film_speaker_dim = speaker_embedding_proj_dim if speaker_embedding_proj_dim > 0 else speaker_embedding_dim

        # Input dim: latent + optional F0 conditioning (concatenated)
        decoder_input_dim = latent_dim + f0_conditioning_dim

        # Defaults for 1D stages
        if intermediate_channels is None:
            # End with output_dim so we can directly reshape to [B, 1, n_mels, T]
            intermediate_channels = [256, 512, 512, 256, output_dim]
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 5, 5, 5]
        if scale_factors is None:
            scale_factors = [2, 2, 1, 1, 1]  # 4x total upsampling

        # Defaults for 2D refinement (small and shallow)
        if conv2d_channels is None:
            conv2d_channels = [64, 32]

        self.scale_factors = scale_factors

        # Activation helpers
        if activation_fn == "snake":
            self.get_activation_1d = lambda c: Snake1d(c)
            self.get_activation_2d = lambda c: Snake2d(c)
        else:
            activation_type = get_activation_type(activation_fn)
            if activation_type in [activations.SwiGLU, activations.Snake]:
                self.get_activation_1d = lambda c: activation_type(c)
                self.get_activation_2d = lambda c: activation_type(c)
            else:
                self.get_activation_1d = lambda _: activation_type()
                self.get_activation_2d = lambda _: activation_type()

        # Speaker embedding projection
        self.speaker_embedding_projection = None
        if speaker_embedding_dim > 0 and speaker_embedding_proj_dim > 0 and speaker_embedding_proj_dim != speaker_embedding_dim:
            self.speaker_embedding_projection = nn.Sequential(
                nn.Linear(speaker_embedding_dim, speaker_embedding_proj_dim),
                nn.SiLU(),
            )

        # === 1D Processing Stages ===
        # Initial projection
        self.initial_conv = nn.Conv1d(decoder_input_dim, intermediate_channels[0], kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(max(1, intermediate_channels[0] // 4), intermediate_channels[0])
        self.initial_act = self.get_activation_1d(intermediate_channels[0])

        # Early FiLM (before any upsampling)
        if speaker_embedding_dim > 0:
            self.early_film_projection = nn.Sequential(
                nn.Linear(self.film_speaker_dim, self.film_speaker_dim, bias=not film_no_bias),
                nn.SiLU(),
                nn.Linear(self.film_speaker_dim, intermediate_channels[0] * 2, bias=not film_no_bias),
            )

        # FiLM projections for each 1D stage
        if speaker_embedding_dim > 0:
            self.speaker_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.film_speaker_dim, self.film_speaker_dim, bias=not film_no_bias),
                    nn.SiLU(),
                    nn.Linear(self.film_speaker_dim, out_c * 2, bias=not film_no_bias),
                )
                for out_c in intermediate_channels
            ])

        # Build 1D stages
        self.stages_1d = nn.ModuleList()
        all_channels = [intermediate_channels[0]] + intermediate_channels

        for in_c, out_c, kernel_size, scale_factor in zip(
            all_channels[:-1], intermediate_channels, kernel_sizes, scale_factors
        ):
            stage = nn.ModuleList()

            # Upsample (if scale_factor > 1)
            if scale_factor > 1:
                stage.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))

            # Conv after upsample
            padding = kernel_size // 2
            stage.append(nn.Conv1d(in_c, out_c, kernel_size, padding=padding))
            stage.append(nn.GroupNorm(max(1, out_c // 4), out_c))
            stage.append(self.get_activation_1d(out_c))

            # Residual blocks
            for _ in range(n_residual_blocks):
                stage.append(ResidualBlock1d(out_c, out_c, kernel_size=kernel_size, activation_fn=activation_fn))

            if dropout > 0:
                stage.append(nn.Dropout(dropout))

            self.stages_1d.append(stage)

        # === 2D Refinement Stage ===
        # Small 2D block at the end for harmonic refinement
        # Input: [B, 1, n_mels, T] (after reshape from 1D output)
        self.conv2d_refinement = nn.ModuleList()

        in_c_2d = 1  # Single channel after reshape
        for out_c_2d in conv2d_channels:
            block = nn.Sequential(
                nn.Conv2d(in_c_2d, out_c_2d, kernel_size=conv2d_kernel_size,
                         padding=(conv2d_kernel_size[0] // 2, conv2d_kernel_size[1] // 2)),
                nn.GroupNorm(max(1, out_c_2d // 4), out_c_2d),
                self.get_activation_2d(out_c_2d),
            )
            self.conv2d_refinement.append(block)

            # Optional residual blocks in 2D
            for _ in range(conv2d_n_residual_blocks):
                self.conv2d_refinement.append(
                    ResidualBlock2d(out_c_2d, out_c_2d, kernel_size=conv2d_kernel_size, activation_fn=activation_fn)
                )

            in_c_2d = out_c_2d

        # Final output conv (2D)
        self.final_conv = nn.Conv2d(conv2d_channels[-1], 1, kernel_size=3, padding=1)

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
        if self.speaker_embedding_dim > 0:
            if hasattr(self, 'early_film_projection'):
                self._init_film_projection(self.early_film_projection)
            for proj in self.speaker_projections:
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
                    if self.zero_init_film_bias:
                        nn.init.zeros_(linear.weight)
                    else:
                        nn.init.normal_(linear.weight, mean=0.0, std=0.02)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)

    def get_output_length(self, input_length: int) -> int:
        """Compute output temporal length given latent length."""
        length = input_length
        for scale_factor in self.scale_factors:
            length = length * scale_factor
        return length

    def forward(
        self,
        z: torch.Tensor,
        speaker_embedding=None,
        f0_embedding: torch.Tensor = None,
        return_film_stats: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim, T'] latent tensor
            speaker_embedding: [B, speaker_dim] or [B, 1, speaker_dim] speaker embedding for FiLM
            f0_embedding: [B, f0_conditioning_dim, T'] optional F0 conditioning (concatenated to z)

        Returns:
            recon: [B, 1, n_mels, T] reconstructed mel spectrogram
        """
        film_stats = {} if return_film_stats else None

        # Process speaker embedding
        if speaker_embedding is not None and self.speaker_embedding_dim > 0:
            if speaker_embedding.dim() == 3:
                speaker_embedding = speaker_embedding.squeeze(1)
            if self.normalize_speaker_embedding:
                speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)
            if self.speaker_embedding_projection is not None:
                speaker_embedding = self.speaker_embedding_projection(speaker_embedding)

        # Concatenate F0 conditioning to latent if provided
        if f0_embedding is not None and self.f0_conditioning_dim > 0:
            if f0_embedding.size(-1) != z.size(-1):
                f0_embedding = F.interpolate(
                    f0_embedding, size=z.size(-1), mode='linear', align_corners=False
                )
            decoder_input = torch.cat([z, f0_embedding], dim=1)
        else:
            decoder_input = z

        # === 1D Processing ===
        # Initial projection
        x = self.initial_conv(decoder_input)
        x = self.initial_norm(x)
        x = self.initial_act(x)

        # Early FiLM conditioning
        if speaker_embedding is not None and self.speaker_embedding_dim > 0:
            film_params = self.early_film_projection(speaker_embedding)
            channels = x.shape[1]
            scale = film_params[:, :channels].unsqueeze(-1)
            shift = film_params[:, channels:].unsqueeze(-1)

            if return_film_stats:
                film_stats["early_scale_raw_mean"] = scale.mean().item()
                film_stats["early_shift_raw_mean"] = shift.mean().item()

            if self.film_scale_bound > 0:
                scale = self.film_scale_bound * torch.tanh(scale)
            if self.film_shift_bound > 0:
                shift = self.film_shift_bound * torch.tanh(shift)

            x = x * (1 + scale) + shift

        # Process through 1D stages with FiLM after each
        for i, stage in enumerate(self.stages_1d):
            for layer in stage:
                x = layer(x)

            # Apply FiLM conditioning after each 1D stage
            if speaker_embedding is not None and self.speaker_embedding_dim > 0:
                film_params = self.speaker_projections[i](speaker_embedding)
                out_c = x.shape[1]
                scale = film_params[:, :out_c].unsqueeze(-1)
                shift = film_params[:, out_c:].unsqueeze(-1)

                if return_film_stats:
                    film_stats[f"stage1d_{i}_scale_raw_mean"] = scale.mean().item()
                    film_stats[f"stage1d_{i}_shift_raw_mean"] = shift.mean().item()

                if self.film_scale_bound > 0:
                    scale = self.film_scale_bound * torch.tanh(scale)
                if self.film_shift_bound > 0:
                    shift = self.film_shift_bound * torch.tanh(shift)

                x = x * (1 + scale) + shift

        # === 1D → 2D Transition ===
        # x is now [B, n_mels, T] - reshape to [B, 1, n_mels, T]
        x = x.unsqueeze(1)

        # === 2D Refinement ===
        for layer in self.conv2d_refinement:
            x = layer(x)

        # Final output
        x = self.final_conv(x)  # [B, 1, n_mels, T]

        if return_film_stats:
            return x, film_stats

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


class GuBERTFeatureToMelSpectrogramVAE(nn.Module):
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
        recon_loss_weight: float = 1.0,
        mse_loss_weight: float = 1.0,
        l1_loss_weight: float = 0.0,
        perceptual_loss_weight: float = 0.1,
        kl_divergence_loss_weight: float = 0.01,
        free_bits: float = 0.0,  # Minimum KL per channel (0 = disabled)
        # Speaker embedding dropout for disentanglement (audio VAE)
        speaker_embedding_dropout: float = 0.0,  # Probability of zeroing speaker embedding during training
        # Instance normalization on latents for speaker disentanglement
        # Removes per-instance statistics (mean/variance) which often encode speaker characteristics
        instance_norm_latents: bool = False,
        # Multi-resolution STFT loss parameters (for audio)
        stft_loss_weight: float = 0.0,  # 0 = disabled
        stft_fft_sizes: list = None,
        stft_hop_sizes: list = None,
        stft_win_lengths: list = None,
        shared_window_buffer=None,  # SharedWindowBuffer instance
        # F0 conditioning parameters (optional)
        f0_predictor: F0Predictor = None,  # Pre-built F0 predictor module
        f0_embedding: F0ConditioningEmbedding = None,  # Pre-built F0 embedding module
        f0_loss_weight: float = 0.1,  # Weight for F0 prediction loss
        vuv_loss_weight: float = 0.1,  # Weight for voiced/unvoiced classification loss
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.recon_loss_weight = recon_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.kl_divergence_loss_weight = kl_divergence_loss_weight
        self.free_bits = free_bits
        self.speaker_embedding_dropout = speaker_embedding_dropout
        self.instance_norm_latents = instance_norm_latents
        self.stft_loss_weight = stft_loss_weight

        # F0 conditioning modules (optional)
        self.f0_predictor = f0_predictor
        self.f0_embedding = f0_embedding
        self.f0_loss_weight = f0_loss_weight
        self.vuv_loss_weight = vuv_loss_weight
        self.use_f0_conditioning = f0_predictor is not None and f0_embedding is not None

        # Multi-resolution STFT loss for audio VAE
        self.stft_loss = None
        if stft_loss_weight > 0 and shared_window_buffer is not None:
            from model.audio.criteria import MultiResolutionSTFTLoss
            self.stft_loss = MultiResolutionSTFTLoss(
                shared_window_buffer=shared_window_buffer,
                fft_sizes=stft_fft_sizes or [256, 512, 1024, 2048],
                hop_sizes=stft_hop_sizes or [64, 128, 256, 512],
                win_lengths=stft_win_lengths or [256, 512, 1024, 2048],
            )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _apply_speaker_embedding_dropout(self, speaker_embedding):
        """
        Apply speaker embedding dropout during training to encourage disentanglement.

        When dropout is applied, the decoder must reconstruct without speaker info,
        forcing it to learn to use the speaker embedding when available (non-dropout steps).
        This prevents the model from encoding all speaker info in the latent space.

        Only applies during training (self.training=True).
        """
        if speaker_embedding is None:
            return None

        if self.training and self.speaker_embedding_dropout > 0:
            # Create a mask that zeros out entire embeddings with probability speaker_embedding_dropout
            # Shape: [B, 1] or [B, 1, 1] to broadcast across embedding dimension
            batch_size = speaker_embedding.shape[0]
            device = speaker_embedding.device

            # Bernoulli mask: 1 = keep, 0 = drop
            keep_mask = torch.rand(batch_size, device=device) >= self.speaker_embedding_dropout

            # Expand mask to match speaker_embedding shape
            if speaker_embedding.dim() == 2:
                keep_mask = keep_mask.unsqueeze(1)  # [B, 1]
            elif speaker_embedding.dim() == 3:
                keep_mask = keep_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

            speaker_embedding = speaker_embedding * keep_mask.float()

        return speaker_embedding

    def encode(self, x, lengths=None) -> tuple:
        """
        Encode input to latent space.

        Returns:
            mu: Latent mean
            logvar: Latent log variance
            learned_speaker_emb (optional): [B, learned_speaker_dim] if encoder.learn_speaker_embedding=True
        """
        # Pass lengths to encoder if it supports it (e.g., AudioVAEEncoder)
        if lengths is not None and hasattr(self.encoder, 'time_strides'):
            return self.encoder(x, lengths=lengths)
        return self.encoder(x)

    def decode(self, z, speaker_embedding=None, f0_embedding=None, features=None, lengths=None, return_film_stats=False) -> torch.Tensor:
        """
        Decode latent to mel spectrogram.

        Args:
            z: Latent tensor
            speaker_embedding: Speaker embedding for FiLM conditioning
            f0_embedding: Pre-computed F0 embedding (optional)
            features: GuBERT features for F0 prediction (optional, used if f0_embedding not provided)
            lengths: Sequence lengths
            return_film_stats: Whether to return FiLM statistics

        If F0 conditioning is enabled and f0_embedding is not provided but features are,
        F0 will be predicted automatically from speaker_embedding + features.
        """
        # Predict F0 if conditioning is enabled but embedding not provided
        if self.use_f0_conditioning and f0_embedding is None and features is not None:
            log_f0_pred, voiced_pred = self.f0_predictor(speaker_embedding, features)
            f0_embedding = self.f0_embedding(log_f0_pred, voiced_pred)

        # Check if decoder supports return_film_stats
        if return_film_stats and hasattr(self.decoder, 'speaker_embedding_dim') and self.decoder.film_speaker_dim > 0:
            return self.decoder(z, speaker_embedding=speaker_embedding, f0_embedding=f0_embedding, return_film_stats=return_film_stats)
        return self.decoder(z, speaker_embedding=speaker_embedding, f0_embedding=f0_embedding)

    def forward(
        self,
        features: torch.Tensor,
        target=None,
        mask=None,
        speaker_embedding=None,
        length=None,
        kl_weight_multiplier: float = 1.0,
        return_film_stats: bool = False,
        use_learned_speaker_embedding: bool = True,
        # F0 supervision (optional, for training F0 predictor)
        target_f0: torch.Tensor = None,  # [B, T] ground truth log F0
        target_voiced: torch.Tensor = None,  # [B, T] ground truth voicing (0 or 1)
        # F0 warmup: use GT F0 instead of predicted F0 for decoder conditioning
        use_gt_f0: bool = False,  # If True, use target_f0/target_voiced for decoder conditioning
    ):
        """
        Forward pass through VAE.

        Args:
            features: [B, D, T'] GuBERT features (channel-first)
            target: [B, n_mels, T] Mel-spectrogram features
            mask: [B, T] optional mask for valid frames
            speaker_embedding: [B, speaker_dim] speaker embedding for FiLM conditioning
            length: [B] optional valid lengths
            kl_weight_multiplier: Multiplier for KL loss (for annealing)
            return_film_stats: If True, return FiLM layer statistics
            use_learned_speaker_embedding: If True and encoder learns speaker, use learned embedding
            target_f0: [B, T] ground truth log F0 for F0 prediction loss (optional)
            target_voiced: [B, T] ground truth voicing mask for VUV loss (optional)
            use_gt_f0: If True, use GT F0 for decoder conditioning instead of predicted F0.
                       F0 predictor is still run for loss computation, but decoder gets GT signal.
                       Useful for warmup to let F0 embedding learn with clean signal.

        Returns:
            recon_x: Reconstructed mel spectrogram
            mu: Latent mean
            logvar: Latent log variance
            losses: Dict with loss components
        """
        encoder_learns_speaker = hasattr(self.encoder, 'learn_speaker_embedding') and self.encoder.learn_speaker_embedding
        encode_result = self.encode(features, lengths=length)

        if encoder_learns_speaker:
            mu, logvar, output_lengths, learned_speaker_emb = encode_result
        else:
            mu, logvar, output_lengths = encode_result
            learned_speaker_emb = None

        z = self.reparameterize(mu, logvar)

        # Apply instance normalization to latents for speaker disentanglement
        # This removes per-instance statistics (mean/variance) which often encode speaker info
        # Speaker characteristics are then re-injected via FiLM conditioning
        if self.instance_norm_latents:
            z = F.instance_norm(z)

        # Determine which speaker embedding to use for decoding
        # If encoder learns speaker embedding and use_learned_speaker_embedding=True, use learned embedding
        # Otherwise, use the provided speaker_embedding (from dataset)
        if encoder_learns_speaker and use_learned_speaker_embedding and learned_speaker_emb is not None:
            decoder_speaker_embedding = learned_speaker_emb
        else:
            # Apply speaker embedding dropout during training for disentanglement
            decoder_speaker_embedding = self._apply_speaker_embedding_dropout(speaker_embedding)

        # F0 prediction and embedding (if enabled)
        f0_emb = None
        log_f0_pred = None
        voiced_pred = None
        if self.use_f0_conditioning:
            # Always predict F0 (for loss computation even during GT warmup)
            log_f0_pred, voiced_pred = self.f0_predictor(decoder_speaker_embedding, features)

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
        decode_result = self.decode(z, speaker_embedding=decoder_speaker_embedding, f0_embedding=f0_emb, return_film_stats=return_film_stats)
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

        if self.free_bits > 0:
            # Free bits: apply minimum KL per channel to prevent posterior collapse
            # Sum over spatial dims, mean over batch -> per-channel KL: [C]
            spatial_dims = list(range(2, mu.dim()))  # [2, 3] for 4D, [2] for 3D
            kl_per_channel = kl_per_element.sum(dim=spatial_dims).mean(dim=0)  # [C]

            # Clamp each channel's KL to at least free_bits
            kl_per_channel = torch.clamp(kl_per_channel, min=self.free_bits)

            # Sum over channels for total KL
            kl_divergence = kl_per_channel.sum()
        else:
            # Original behavior: sum over all latent dims, mean over batch
            latent_dims = list(range(1, mu.dim()))  # [1, 2, 3] for 4D, [1, 2] for 3D
            kl_divergence = kl_per_element.sum(dim=latent_dims).mean()

        # Reconstruction losses (with optional masking)
        if mask is not None:
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
            if self.l1_loss_weight > 0:
                abs_error = torch.abs(recon_x - target)
                masked_abs_error = abs_error * mask_expanded
                l1_loss = (masked_abs_error.sum(dim=list(range(1, masked_abs_error.dim()))) / (valid_elements + 1e-5)).mean()
            else:
                l1_loss = torch.tensor(0.0, device=target.device)
        else:
            # Standard unmasked losses
            mse_loss = F.mse_loss(recon_x, target, reduction='mean')
            l1_loss = F.l1_loss(recon_x, target, reduction='mean') if self.l1_loss_weight > 0 else torch.tensor(0.0, device=target.device)

        recon_loss = self.mse_loss_weight * mse_loss + self.l1_loss_weight * l1_loss

        # Multi-resolution STFT loss (for audio, requires waveform data passed separately)
        # This is computed externally when waveforms are available
        stft_loss = torch.tensor(0.0, device=target.device)

        # Apply KL weight multiplier for KL annealing
        effective_kl_weight = self.kl_divergence_loss_weight * kl_weight_multiplier

        total_loss = (
            self.recon_loss_weight * recon_loss
            + effective_kl_weight * kl_divergence
            + self.stft_loss_weight * stft_loss
        )

        # F0 prediction losses (if F0 conditioning is enabled and ground truth provided)
        f0_loss = torch.tensor(0.0, device=target.device)
        vuv_loss = torch.tensor(0.0, device=target.device)
        if self.use_f0_conditioning and log_f0_pred is not None and target_f0 is not None and target_voiced is not None:
            # Upsample predictions to match target resolution if needed
            if log_f0_pred.size(-1) != target_f0.size(-1):
                log_f0_pred_up = F.interpolate(
                    log_f0_pred.unsqueeze(1), size=target_f0.size(-1), mode='linear', align_corners=False
                ).squeeze(1)
                voiced_pred_up = F.interpolate(
                    voiced_pred.unsqueeze(1), size=target_f0.size(-1), mode='linear', align_corners=False
                ).squeeze(1)
            else:
                log_f0_pred_up = log_f0_pred
                voiced_pred_up = voiced_pred

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
            total_loss = total_loss + self.f0_loss_weight * f0_loss + self.vuv_loss_weight * vuv_loss

        losses = {
            "total_loss": total_loss,
            "kl_divergence": kl_divergence,
            "recon_loss": recon_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "stft_loss": stft_loss,
            "f0_loss": f0_loss,
            "vuv_loss": vuv_loss,
            "kl_weight_multiplier": torch.tensor(kl_weight_multiplier, device=target.device),
        }

        # Add F0 predictions for external logging/monitoring
        if self.use_f0_conditioning and log_f0_pred is not None:
            losses["log_f0_pred"] = log_f0_pred
            losses["voiced_pred"] = voiced_pred

        # Add FiLM statistics if requested
        if return_film_stats and film_stats is not None:
            losses["film_stats"] = film_stats

        # Add learned speaker embedding to losses dict for external use (e.g., GRL speaker ID loss)
        if learned_speaker_emb is not None:
            losses["learned_speaker_embedding"] = learned_speaker_emb

        return recon_x, mu, logvar, losses

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
    "tiny_test": lambda latent_channels, speaker_embedding_dim=0, speaker_embedding_proj_dim=0, normalize_speaker_embedding=False, film_scale_bound=0.0, film_shift_bound=0.0, zero_init_film_bias=True, film_no_bias=False, learn_speaker_embedding=False, learned_speaker_dim=256, use_input_instance_norm=False, **kwargs: VAE(
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
            use_instance_norm=use_input_instance_norm,
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
    "medium_no_attn": lambda latent_channels, speaker_embedding_dim=0, speaker_embedding_proj_dim=0, normalize_speaker_embedding=False, film_scale_bound=0.0, film_shift_bound=0.0, zero_init_film_bias=True, film_no_bias=False, learn_speaker_embedding=False, learned_speaker_dim=256, use_input_instance_norm=False, **kwargs: VAE(
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
            use_instance_norm=use_input_instance_norm,
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
    "large": lambda latent_channels, speaker_embedding_dim=0, speaker_embedding_proj_dim=0, normalize_speaker_embedding=False, film_scale_bound=0.0, film_shift_bound=0.0, zero_init_film_bias=True, film_no_bias=False, learn_speaker_embedding=False, learned_speaker_dim=256, use_input_instance_norm=False, **kwargs: VAE(
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
            use_instance_norm=use_input_instance_norm,
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
    "tiny_mels": lambda input_dim=128, latent_dim=16, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[128],
            kernel_sizes=[3],
            strides=[2],  # 2x downsampling
            n_residual_blocks=0,
            activation_fn="silu",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=80,  # Mel-spectrogram dimension
            intermediate_channels=[32, 32, 32],
            kernel_sizes=[3, 5, 5],
            scale_factors=[2, 2, 2],  # 8x upsampling (REQUIRED FOR MELS FROM GUBERT, WHICH ARE 4X DOWNSAMPLED -> 2X DOWNSAMPLED IN ENCODER)
            n_residual_blocks=0,
            activation_fn="silu",
            dropout=0.1,
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
    # ~3.3M parameters: ~1.8M encoder, ~1.5M decoder
    "small_mels": lambda input_dim=128, latent_dim=16, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[512],
            kernel_sizes=[3],
            strides=[2],  # 2x downsampling
            n_residual_blocks=1,
            activation_fn="silu",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=80,  # Mel-spectrogram dimension
            intermediate_channels=[128, 256, 512],
            kernel_sizes=[3, 5, 5],
            scale_factors=[2, 2, 2],  # 8x upsampling (REQUIRED FOR MELS FROM GUBERT, WHICH ARE 4X DOWNSAMPLED -> 2X DOWNSAMPLED IN ENCODER)
            n_residual_blocks=0,
            activation_fn="silu",
            dropout=0.1,
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
    # ~9.9M parameters: ~5M encoder, ~4.9M decoder
    "medium_mels": lambda input_dim=128, latent_dim=16, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[512],
            kernel_sizes=[3],
            strides=[2],  # 2x downsampling
            n_residual_blocks=3,
            activation_fn="silu",
            dropout=0.0,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=80,  # Mel-spectrogram dimension
            intermediate_channels=[128, 256, 512],
            kernel_sizes=[3, 5, 5],
            scale_factors=[2, 2, 2],  # 8x upsampling (REQUIRED FOR MELS FROM GUBERT, WHICH ARE 4X DOWNSAMPLED -> 2X DOWNSAMPLED IN ENCODER)
            n_residual_blocks=1,
            activation_fn="silu",
            dropout=0.0,
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
    # ~21M parameters: ~11M encoder, ~10M decoder
    "large_mels": lambda input_dim=128, latent_dim=16, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[768],
            kernel_sizes=[3],
            strides=[2],  # 2x downsampling
            n_residual_blocks=3,
            activation_fn="silu",
            dropout=0.0,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=80,  # Mel-spectrogram dimension
            intermediate_channels=[192, 384, 768],
            kernel_sizes=[3, 5, 5],
            scale_factors=[2, 2, 2],  # 8x upsampling (REQUIRED FOR MELS FROM GUBERT, WHICH ARE 4X DOWNSAMPLED -> 2X DOWNSAMPLED IN ENCODER)
            n_residual_blocks=1,
            activation_fn="silu",
            dropout=0.0,
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

    # =============================================================================
    # F0-conditioned configs (with explicit pitch prediction)
    # These configs add F0 prediction from speaker embedding + GuBERT features,
    # which is concatenated to the latent before decoding.
    # =============================================================================

    "small_mels_f0_2d": lambda input_dim=128, latent_dim=16, f0_embedding_dim=64, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, f0_loss_weight=0.1, vuv_loss_weight=0.1, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[256],
            kernel_sizes=[3],
            strides=[1],  # No downsampling - preserve full temporal resolution
            n_residual_blocks=4,
            activation_fn="silu",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder2D(
            latent_dim=latent_dim,
            output_dim=80,  # n_mels
            # 1D stage: process latent temporally
            conv1d_channels=320,  # 320 = 32 channels × 10 freq bins
            conv1d_n_residual_blocks=2,
            conv1d_kernel_size=5,
            conv1d_upsample_factor=1,  # No 1D upsampling, all upsampling in 2D
            # 1D→2D transition
            initial_freq_bins=10,  # Start with 10 freq bins
            # 2D stages: upsample freq 10→20→40→80, time T'→2T'→4T'
            intermediate_channels_2d=[64, 96, 144],
            kernel_sizes_2d=[(3, 5), (3, 5), (3, 5)],
            scale_factors_2d=[(2, 2), (2, 2), (2, 1)],  # 8x freq, 4x time
            n_residual_blocks_2d=2,
            activation_fn="silu",
            dropout=0.1,
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
            f0_conditioning_dim=f0_embedding_dim,
        ),
        f0_predictor=F0Predictor(
            speaker_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            gubert_dim=input_dim,
            hidden_dim=384,
            n_layers=4,
            kernel_size=5,
        ),
        f0_embedding=F0ConditioningEmbedding(
            embedding_dim=f0_embedding_dim,
            n_harmonics=8,
        ),
        f0_loss_weight=f0_loss_weight,
        vuv_loss_weight=vuv_loss_weight,
        **kwargs
    ),

    "small_mels_f0_harmonics_2d": lambda input_dim=128, latent_dim=16, f0_embedding_dim=64, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, f0_loss_weight=0.1, vuv_loss_weight=0.1, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[256],
            kernel_sizes=[3],
            strides=[1],  # No downsampling - preserve full temporal resolution
            n_residual_blocks=4,
            activation_fn="silu",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder2D(
            latent_dim=latent_dim,
            output_dim=80,  # n_mels
            # 1D stage: process latent temporally
            conv1d_channels=320,  # 320 = 32 channels × 10 freq bins
            conv1d_n_residual_blocks=2,
            conv1d_kernel_size=5,
            conv1d_upsample_factor=1,  # No 1D upsampling, all upsampling in 2D
            # 1D→2D transition
            initial_freq_bins=10,  # Start with 10 freq bins
            # 2D stages: upsample freq 10→20→40→80, time T'→2T'→4T'
            intermediate_channels_2d=[64, 96, 144],
            kernel_sizes_2d=[(3, 5), (3, 5), (3, 5)],
            scale_factors_2d=[(2, 2), (2, 2), (2, 1)],  # 8x freq, 4x time
            n_residual_blocks_2d=2,
            activation_fn="silu",
            dropout=0.1,
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
            f0_conditioning_dim=f0_embedding_dim,
        ),
        f0_predictor=F0Predictor(
            speaker_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            gubert_dim=input_dim,
            hidden_dim=384,
            n_layers=4,
            kernel_size=5,
        ),
        f0_embedding=F0ConditioningEmbedding(
            embedding_dim=f0_embedding_dim,
            n_harmonics=8,
            use_harmonic_sinusoids=True,
        ),
        f0_loss_weight=f0_loss_weight,
        vuv_loss_weight=vuv_loss_weight,
        **kwargs
    ),

    # Small mels with F0 conditioning
    "small_mels_f0": lambda input_dim=128, latent_dim=16, f0_embedding_dim=64, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, f0_loss_weight=0.1, vuv_loss_weight=0.1, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[512],
            kernel_sizes=[3],
            strides=[2],  # 2x downsampling
            n_residual_blocks=1,
            activation_fn="silu",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=80,  # Mel-spectrogram dimension
            intermediate_channels=[128, 256, 512],
            kernel_sizes=[3, 5, 5],
            scale_factors=[2, 2, 2],  # 8x upsampling
            n_residual_blocks=0,
            activation_fn="silu",
            dropout=0.1,
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
            f0_conditioning_dim=f0_embedding_dim,  # F0 embedding concatenated to latent
        ),
        f0_predictor=F0Predictor(
            speaker_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            gubert_dim=input_dim,
            hidden_dim=256,
            n_layers=3,
            kernel_size=5,
        ),
        f0_embedding=F0ConditioningEmbedding(
            embedding_dim=f0_embedding_dim,
            n_harmonics=6,
        ),
        f0_loss_weight=f0_loss_weight,
        vuv_loss_weight=vuv_loss_weight,
        **kwargs
    ),

    # Medium mels with F0 conditioning
    "medium_mels_f0": lambda input_dim=128, latent_dim=16, f0_embedding_dim=64, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, f0_loss_weight=0.1, vuv_loss_weight=0.1, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[512],
            kernel_sizes=[3],
            strides=[2],  # 2x downsampling
            n_residual_blocks=3,
            activation_fn="silu",
            dropout=0.0,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=80,  # Mel-spectrogram dimension
            intermediate_channels=[128, 256, 512],
            kernel_sizes=[3, 5, 5],
            scale_factors=[2, 2, 2],  # 8x upsampling
            n_residual_blocks=1,
            activation_fn="silu",
            dropout=0.0,
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
            f0_conditioning_dim=f0_embedding_dim,
        ),
        f0_predictor=F0Predictor(
            speaker_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            gubert_dim=input_dim,
            hidden_dim=256,
            n_layers=3,
            kernel_size=5,
        ),
        f0_embedding=F0ConditioningEmbedding(
            embedding_dim=f0_embedding_dim,
            n_harmonics=6,
        ),
        f0_loss_weight=f0_loss_weight,
        vuv_loss_weight=vuv_loss_weight,
        **kwargs
    ),

    # Large mels with F0 conditioning - no encoder downsampling variant
    # Keeps full temporal resolution in latent (16 x T instead of 16 x T/2)
    # Better for preserving fine harmonic structure and voicing details
    # ~17.8M parameters, ~6.5M encoder, ~8M decoder, ~3.3M F0 predictor + embedding
    "large_mels_f0_no_downsample": lambda input_dim=128, latent_dim=32, f0_embedding_dim=64, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, f0_loss_weight=0.1, vuv_loss_weight=0.1, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[512],
            kernel_sizes=[3],
            strides=[1],  # No downsampling - preserve full temporal resolution
            n_residual_blocks=4,
            activation_fn="silu",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=80,  # Mel-spectrogram dimension
            intermediate_channels=[256, 512],
            kernel_sizes=[5, 5],
            scale_factors=[2, 2],  # 4x upsampling (GuBERT rate -> mel rate)
            n_residual_blocks=2,
            activation_fn="silu",
            dropout=0.1,
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
            f0_conditioning_dim=f0_embedding_dim,
        ),
        f0_predictor=F0Predictor(
            speaker_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            gubert_dim=input_dim,
            hidden_dim=384,
            n_layers=4,
            kernel_size=5,
        ),
        f0_embedding=F0ConditioningEmbedding(
            embedding_dim=f0_embedding_dim,
            n_harmonics=8,
        ),
        f0_loss_weight=f0_loss_weight,
        vuv_loss_weight=vuv_loss_weight,
        **kwargs
    ),

    # Large mels with F0 conditioning - hybrid 1D→2D decoder
    # No encoder downsampling, decoder transitions from 1D to 2D for proper mel generation
    # 1D stage processes temporal dynamics, 2D stages generate frequency-local patterns (harmonics)
    "large_mels_f0_2d": lambda input_dim=128, latent_dim=16, f0_embedding_dim=64, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, f0_loss_weight=0.1, vuv_loss_weight=0.1, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[512],
            kernel_sizes=[3],
            strides=[1],  # No downsampling - preserve full temporal resolution
            n_residual_blocks=4,
            activation_fn="silu",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder2D(
            latent_dim=latent_dim,
            output_dim=80,  # n_mels
            # 1D stage: process latent temporally
            conv1d_channels=320,  # 320 = 32 channels × 10 freq bins
            conv1d_n_residual_blocks=2,
            conv1d_kernel_size=5,
            conv1d_upsample_factor=1,  # No 1D upsampling, all upsampling in 2D
            # 1D→2D transition
            initial_freq_bins=10,  # Start with 10 freq bins
            # 2D stages: upsample freq 10→20→40→80, time T'→2T'→4T'
            intermediate_channels_2d=[64, 128, 256],
            kernel_sizes_2d=[(3, 5), (3, 5), (3, 5)],
            scale_factors_2d=[(2, 2), (2, 2), (2, 1)],  # 8x freq, 4x time
            n_residual_blocks_2d=2,
            activation_fn="silu",
            dropout=0.1,
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
            f0_conditioning_dim=f0_embedding_dim,
        ),
        f0_predictor=F0Predictor(
            speaker_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            gubert_dim=input_dim,
            hidden_dim=384,
            n_layers=4,
            kernel_size=5,
        ),
        f0_embedding=F0ConditioningEmbedding(
            embedding_dim=f0_embedding_dim,
            n_harmonics=8,
        ),
        f0_loss_weight=f0_loss_weight,
        vuv_loss_weight=vuv_loss_weight,
        **kwargs
    ),

    # Large mels with F0 conditioning - mostly 1D decoder with 2D refinement at the end
    # Preserves temporal dynamics better (for content/phonemes), uses 2D only for final harmonic refinement
    # FiLM speaker conditioning applied throughout the 1D stages (not just at 2D transition)
    "large_mels_f0_mostly1d": lambda input_dim=128, latent_dim=16, f0_embedding_dim=64, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, f0_loss_weight=0.1, vuv_loss_weight=0.1, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[512],
            kernel_sizes=[3],
            strides=[1],  # No downsampling - preserve full temporal resolution
            n_residual_blocks=4,
            activation_fn="silu",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoderMostly1D(
            latent_dim=latent_dim,
            output_dim=80,  # n_mels
            # 1D stages: do all the heavy lifting (upsampling, content)
            # End with n_mels channels so we can reshape to [B, 1, n_mels, T]
            intermediate_channels=[256, 512, 512, 256, 80],
            kernel_sizes=[5, 5, 5, 5, 5],
            scale_factors=[2, 2, 1, 1, 1],  # 4x upsampling in 1D
            n_residual_blocks=2,
            # 2D refinement: small block at the end for harmonic sharpening
            conv2d_channels=[64, 32],
            conv2d_kernel_size=(3, 5),
            conv2d_n_residual_blocks=1,
            activation_fn="silu",
            dropout=0.1,
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
            f0_conditioning_dim=f0_embedding_dim,
        ),
        f0_predictor=F0Predictor(
            speaker_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            gubert_dim=input_dim,
            hidden_dim=384,
            n_layers=4,
            kernel_size=5,
        ),
        f0_embedding=F0ConditioningEmbedding(
            embedding_dim=f0_embedding_dim,
            n_harmonics=8,
        ),
        f0_loss_weight=f0_loss_weight,
        vuv_loss_weight=vuv_loss_weight,
        **kwargs
    ),

    # Large mels with F0 conditioning
    "large_mels_f0": lambda input_dim=128, latent_dim=32, f0_embedding_dim=64, speaker_embedding_dim=192, learned_speaker_dim=192, learn_speaker_embedding=False, speaker_embedding_proj_dim=192, normalize_speaker_embedding=True, film_scale_bound=0.1, film_shift_bound=0.1, zero_init_film_bias=True, film_no_bias=False, f0_loss_weight=0.1, vuv_loss_weight=0.1, **kwargs: GuBERTFeatureToMelSpectrogramVAE(
        encoder=GuBERTFeatureVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_channels=[512],
            kernel_sizes=[3],
            strides=[2],  # 2x downsampling
            n_residual_blocks=4,
            activation_fn="silu",
            dropout=0.1,
        ),
        decoder=GuBERTFeatureVAEDecoder(
            latent_dim=latent_dim,
            output_dim=80,  # Mel-spectrogram dimension
            intermediate_channels=[192, 384, 768],
            kernel_sizes=[3, 5, 5],
            scale_factors=[2, 2, 2],  # 8x upsampling
            n_residual_blocks=2,
            activation_fn="silu",
            dropout=0.1,
            speaker_embedding_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            speaker_embedding_proj_dim=speaker_embedding_proj_dim if not learn_speaker_embedding else 0,
            normalize_speaker_embedding=normalize_speaker_embedding,
            film_scale_bound=film_scale_bound,
            film_shift_bound=film_shift_bound,
            zero_init_film_bias=zero_init_film_bias,
            film_no_bias=film_no_bias,
            f0_conditioning_dim=f0_embedding_dim,
        ),
        f0_predictor=F0Predictor(
            speaker_dim=learned_speaker_dim if learn_speaker_embedding else speaker_embedding_dim,
            gubert_dim=input_dim,
            hidden_dim=384,
            n_layers=4,
            kernel_size=5,
        ),
        f0_embedding=F0ConditioningEmbedding(
            embedding_dim=f0_embedding_dim,
            n_harmonics=8,
        ),
        f0_loss_weight=f0_loss_weight,
        vuv_loss_weight=vuv_loss_weight,
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
