import torch
import torch.nn as nn
import torch.nn.functional as F

from model import activations
from model.vae import VAE
from utils.model_utils import get_activation_type


class ImageBottleneckAttention(nn.Module):
    """
    2D self-attention module for the VAE bottleneck with learned positional embeddings.

    Flattens the H×W spatial grid into a sequence, applies multi-head self-attention,
    and reshapes back. Uses learned 2D positional embeddings (separate for x and y)
    that are added together.

    For a bottleneck with H=8, W=8, this creates a 64-token sequence - very tractable.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_height: int = 64,
        max_width: int = 64,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        # Pre-norm
        self.norm = nn.GroupNorm(max(1, channels // 4), channels)

        # QKV projections
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        # Learned 2D positional embeddings (separate for height and width, added together)
        self.pos_embed_h = nn.Parameter(torch.zeros(1, max_height, channels))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, max_width, channels))

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        # Initialize projections
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

        # Initialize output projection with smaller weights for stable residual
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed_h, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_w, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        """
        Args:
            x: [B, C, H, W] input tensor
            return_attention_weights: If True, also return attention weights

        Returns:
            out: [B, C, H, W] output tensor
            attn_weights (optional): [B, num_heads, H*W, H*W] if return_attention_weights=True
        """
        B, C, H, W = x.shape
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Reshape to sequence: [B, C, H, W] -> [B, H*W, C]
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Add 2D positional embeddings
        # Create position indices for each spatial location
        h_pos = self.pos_embed_h[:, :H, :]  # [1, H, C]
        w_pos = self.pos_embed_w[:, :W, :]  # [1, W, C]

        # Broadcast and add: each position (i,j) gets pos_embed_h[i] + pos_embed_w[j]
        # h_pos: [1, H, 1, C], w_pos: [1, 1, W, C] -> [1, H, W, C] -> [1, H*W, C]
        pos_embed = (h_pos.unsqueeze(2) + w_pos.unsqueeze(1)).reshape(1, H * W, C)
        x = x + pos_embed

        # QKV projections
        q = self.q_proj(x)  # [B, H*W, C]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: [B, H*W, C] -> [B, num_heads, H*W, head_dim]
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, H*W, H*W]
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(attn_weights.dtype)
        attn_weights_dropped = self.dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights_dropped, v)  # [B, num_heads, H*W, head_dim]

        # Reshape back: [B, num_heads, H*W, head_dim] -> [B, H*W, C]
        attn_out = attn_out.transpose(1, 2).reshape(B, H * W, C)

        # Output projection
        out = self.out_proj(attn_out)

        # Reshape back to spatial: [B, H*W, C] -> [B, C, H, W]
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Residual connection
        out = residual + out

        if return_attention_weights:
            return out, attn_weights

        return out


class ImageVAEEncoder(nn.Module):
    """
    Image VAE encoder with optional bottleneck attention.

    Args:
        in_channels: Number of input channels (3 for RGB)
        latent_channels: Number of latent channels
        intermediate_channels: List of channel counts for each downsampling stage
        activation_fn: Activation function name
        use_attention: Whether to use bottleneck attention
        attention_heads: Number of attention heads (if use_attention=True)
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        intermediate_channels: list = [32, 64, 128],
        activation_fn: str = "silu",
        use_attention: bool = False,
        attention_heads: int = 4,
        logvar_clamp_max: float = 4.0,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.logvar_clamp_max = logvar_clamp_max

        activation_type = get_activation_type(activation_fn)
        if activation_type not in [activations.SwiGLU, activations.Snake]:
            activation = lambda _: activation_type()  # drop unused arg
        else:
            activation = activation_type

        channels = [in_channels] + intermediate_channels

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c)
            )
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        # Bottleneck attention (on smallest resolution, after all conv stages)
        if use_attention:
            self.attention = ImageBottleneckAttention(
                channels=intermediate_channels[-1],
                num_heads=attention_heads,
            )

        self.fc_mu = nn.Conv2d(channels[-1], latent_channels, kernel_size=3, padding=1)
        self.fc_logvar = nn.Conv2d(channels[-1], latent_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module in [self.fc_mu, self.fc_logvar]:
                    continue  # Skip mu/logvar, handled below
                if isinstance(module, nn.Conv2d):
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

    def forward(self, x, return_attention_weights: bool = False, **kwargs):
        """
        Args:
            x: [B, C, H, W] input image
            return_attention_weights: If True and use_attention=True, return attention weights
            **kwargs: Ignored (for compatibility with audio VAE which uses speaker_embedding, lengths)

        Returns:
            mu: [B, latent_channels, H', W'] latent mean
            logvar: [B, latent_channels, H', W'] latent log variance
            attn_weights (optional): [B, num_heads, H'*W', H'*W'] if return_attention_weights=True
        """
        for upsample in self.channel_upsample:
            x = upsample(x)

        # Apply bottleneck attention
        attn_weights = None
        if self.use_attention:
            attn_result = self.attention(x, return_attention_weights=return_attention_weights)
            if return_attention_weights:
                x, attn_weights = attn_result
            else:
                x = attn_result

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10.0, max=self.logvar_clamp_max)

        if return_attention_weights:
            return mu, logvar, attn_weights

        return mu, logvar


class ImageVAEDecoder(nn.Module):
    """
    Image VAE decoder with optional bottleneck attention.

    Args:
        latent_channels: Number of latent channels
        out_channels: Number of output channels (3 for RGB)
        intermediate_channels: List of channel counts for each upsampling stage
        activation_fn: Activation function name
        use_attention: Whether to use bottleneck attention
        attention_heads: Number of attention heads (if use_attention=True)
    """
    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        intermediate_channels: list = [128, 64, 32],
        activation_fn: str = "silu",
        use_attention: bool = False,
        attention_heads: int = 4,
        use_final_tanh: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention

        activation_type = get_activation_type(activation_fn)
        if activation_type not in [activations.SwiGLU, activations.Snake]:
            activation = lambda _: activation_type()  # drop unused arg
        else:
            activation = activation_type

        channels = [latent_channels] + intermediate_channels

        # Initial projection from latent space (needed for attention to work on proper channels)
        if use_attention:
            self.initial_conv = nn.Conv2d(latent_channels, intermediate_channels[0], kernel_size=3, padding=1)
            self.initial_norm = nn.GroupNorm(max(1, intermediate_channels[0] // 4), intermediate_channels[0])
            self.initial_act = activation(intermediate_channels[0])

            # Bottleneck attention (before upsampling, at smallest resolution)
            self.attention = ImageBottleneckAttention(
                channels=intermediate_channels[0],
                num_heads=attention_heads,
            )

            # Adjust channels list for upsampling stages (skip first projection)
            channels = [intermediate_channels[0]] + intermediate_channels

        self.channel_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.GroupNorm(out_c // 4, out_c),
                activation(out_c),
            )
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.final_conv = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)
        if use_final_tanh:
            self.final_act = nn.Tanh()
        else:
            self.final_act = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z, *_, return_attention_weights: bool = False, **__):
        """
        Args:
            z: [B, latent_channels, H, W] latent tensor
            return_attention_weights: If True and use_attention=True, return attention weights

        Returns:
            recon_x: [B, out_channels, H', W'] reconstructed image
            attn_weights (optional): [B, num_heads, H*W, H*W] if return_attention_weights=True
        """
        # speaker_embedding is ignored for image VAE (only used for audio VAE)

        attn_weights = None
        if self.use_attention:
            # Initial projection to intermediate channels
            z = self.initial_conv(z)
            z = self.initial_norm(z)
            z = self.initial_act(z)

            # Apply bottleneck attention
            attn_result = self.attention(z, return_attention_weights=return_attention_weights)
            if return_attention_weights:
                z, attn_weights = attn_result
            else:
                z = attn_result

        for upsample in self.channel_upsample:
            z = upsample(z)

        recon_x = self.final_conv(z)
        recon_x = self.final_act(recon_x)

        if return_attention_weights:
            return recon_x, attn_weights

        return recon_x


model_config_lookup = {
    "tiny": lambda latent_channels, **kwargs: VAE(
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[32, 64],
            activation_fn="silu"
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[64, 32],
            activation_fn="silu"
        ),
        **kwargs
    ),
    # 200,000 total params (~100K encoder, ~100K decoder)
    "mini": lambda latent_channels, **kwargs: VAE(
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[32, 64, 128],
            activation_fn="silu"
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[128, 64, 32],
            activation_fn="silu"
        ),
        **kwargs
    ),
    # ~771K total params (~390K encoder, ~381K decoder)
    "small": lambda latent_channels, **kwargs: VAE(
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[64, 128, 256],
            activation_fn="silu"
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[256, 128, 64],
            activation_fn="silu"
        ),
        **kwargs
    ),
    # ~3M total params (~1.5M encoder, ~1.5M decoder)
    "medium": lambda latent_channels, **kwargs: VAE(
        # creates 32x32 latents for 256x256 images
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[320, 480, 640],
            activation_fn="silu"
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[640, 480, 320],
            activation_fn="silu",
        ),
        **kwargs
    ),
    "medium_stability_test": lambda latent_channels, **kwargs: VAE(
        # creates 32x32 latents for 256x256 images
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[320, 480, 640],
            activation_fn="silu"
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[640, 480, 320],
            activation_fn="silu",
            use_final_tanh=True,
        ),
        **kwargs
    ),
    # =========================================================================
    # Attention-enabled configs
    # These add bottleneck attention for improved global coherence
    # =========================================================================
    # ~250K total params - mini with attention
    "mini_attn": lambda latent_channels, **kwargs: VAE(
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[32, 64, 128],
            activation_fn="silu",
            use_attention=True,
            attention_heads=4,
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[128, 64, 32],
            activation_fn="silu",
            use_attention=True,
            attention_heads=4,
        ),
        **kwargs
    ),
    # ~900K total params - small with attention
    "small_attn": lambda latent_channels, **kwargs: VAE(
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[64, 128, 256],
            activation_fn="silu",
            use_attention=True,
            attention_heads=4,
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[256, 128, 64],
            activation_fn="silu",
            use_attention=True,
            attention_heads=4,
        ),
        **kwargs
    ),
    # ~3.5M total params - medium with attention
    "medium_attn": lambda latent_channels, **kwargs: VAE(
        encoder=ImageVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            intermediate_channels=[320, 480, 640],
            activation_fn="silu",
            use_attention=True,
            attention_heads=8,
        ),
        decoder=ImageVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            intermediate_channels=[640, 480, 320],
            activation_fn="silu",
            use_attention=True,
            attention_heads=8,
        ),
        **kwargs
    ),
}
