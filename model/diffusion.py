import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from functools import partial
from typing import Optional, Union

from torch.amp import autocast

from model import activations, norms, SinusoidalPositionEmbeddings
from utils import configuration
from utils.model_utils import get_activation_type


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, activation, norm):
        super().__init__()

        activation_type = get_activation_type(activation)

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = norm(out_channels)
        self.act = activation_type() if activation_type is not activations.SwiGLU else activations.SwiGLU(in_channels)

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor]=None):
        # x: [B, C, H, W]
        # C = hidden_size/hidden_channels
        # H = n_mels for audio, height for image
        # W = time_frames for audio, width for image

        x = self.proj(x)
        # switch channel to last dimension (now [B, H, W, C])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)

        if time_embedding is not None:
            # x is [B, H, W, C]
            # time_embedding is [B, C]
            x = x + time_embedding[:, None, None, :]

        # switch back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, activation, norm):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim

        activation_type = get_activation_type(activation)

        if time_embedding_dim is not None:
            self.time_mlp = nn.Sequential(
                activation_type() if activation_type is not activations.SwiGLU else activations.SwiGLU(time_embedding_dim),
                nn.Linear(time_embedding_dim, out_channels),
            )

        self.block1 = Block(in_channels, out_channels, activation, norm=norm)
        self.block2 = Block(out_channels, out_channels, activation, norm=norm)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.GroupNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(self.shortcut, nn.Conv2d):
            nn.init.kaiming_normal_(self.shortcut.weight, a=0.2)
            self.shortcut.bias.data.zero_()

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor]=None) -> torch.Tensor:
        if self.time_embedding_dim is not None and time_embedding is not None:
            time_embedding = self.time_mlp(time_embedding)

        h = self.block1(x, time_embedding=time_embedding)
        h = self.block2(h)
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(
            self,
            stride,
            in_channels: int,
            out_channels: int,
            time_embedding_dim: int,
            activation,
            self_attn_class,
            cross_attn_class,
            norm_class,
            has_attn: bool=False,
            has_condition: bool=False,
            context_dim: Optional[int]=None,
            num_res_blocks: int=2,
            dropout_p: float=0.1,
            self_attn_n_heads=6,
            self_attn_d_queries=64,
            self_attn_d_values=64,
            self_attn_use_flash_attention=True,
            cross_attn_n_heads=6,
            cross_attn_d_queries=64,
            cross_attn_d_values=64,
            cross_attn_use_flash_attention=True
    ):
        super().__init__()

        self.norms = nn.ModuleList([
            norms.RMSNorm(in_channels if i == 0 else out_channels)
            for i in range(num_res_blocks)
        ])

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, time_embedding_dim, activation, norm_class)
            for i in range(num_res_blocks)
        ])

        self.attn_blocks = nn.ModuleList([
            self_attn_class(
                out_channels, self_attn_n_heads, self_attn_d_queries, self_attn_d_values, use_flash_attention=self_attn_use_flash_attention, dropout_p=dropout_p
            ) if has_attn else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self.cross_attn_blocks = nn.ModuleList([
            cross_attn_class(
                out_channels, cross_attn_n_heads, cross_attn_d_queries, cross_attn_d_values, context_dim=context_dim, use_flash_attention=cross_attn_use_flash_attention, dropout_p=dropout_p
            ) if has_attn and has_condition else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.downsample.weight, a=0.2)
        self.downsample.bias.data.zero_()

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor]=None, condition=None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, H, W]
        for norm, res_block, attn_block, cross_attn_block in zip(self.norms, self.res_blocks, self.attn_blocks, self.cross_attn_blocks):
            # switch channel to last dimension (now [B, H, W, C])
            x = x.permute(0, 2, 3, 1).contiguous()
            x = norm(x)
            # switch back to [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
            x = res_block(x, time_embedding)
            x = x + attn_block(x)
            if condition is not None and not isinstance(cross_attn_block, nn.Identity):
                x = x + cross_attn_block(x, condition)
        return self.downsample(x), x

class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_embedding_dim: int,
                 activation,
                 self_attn_class,
                 cross_attn_class,
                 norm_class,
                 has_attn: bool=False,
                 has_condition: bool=False,
                 context_dim: Optional[int]=None,
                 num_res_blocks: int=2,
                 dropout_p: float=0.1,
                 self_attn_n_heads=6,
                 self_attn_d_queries=64,
                 self_attn_d_values=64,
                 self_attn_use_flash_attention=True,
                 cross_attn_n_heads=6,
                 cross_attn_d_queries=64,
                 cross_attn_d_values=64,
                 cross_attn_use_flash_attention=True):
        super().__init__()

        self.upsample_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.norms = nn.ModuleList([
            norms.RMSNorm(in_channels*2 if i == 0 else out_channels)
            for i in range(num_res_blocks)
        ])

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels*2 if i == 0 else out_channels, out_channels, time_embedding_dim, activation, norm_class)
            for i in range(num_res_blocks)
        ])

        self.attn_blocks = nn.ModuleList([
            self_attn_class(
                out_channels, self_attn_n_heads, self_attn_d_queries, self_attn_d_values, use_flash_attention=self_attn_use_flash_attention, dropout_p=dropout_p
            ) if has_attn else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self.cross_attn_blocks = nn.ModuleList([
            cross_attn_class(
                out_channels, cross_attn_n_heads, cross_attn_d_queries, cross_attn_d_values, context_dim=context_dim, use_flash_attention=cross_attn_use_flash_attention, dropout_p=dropout_p
            ) if has_attn and has_condition else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.upsample_conv.weight, a=0.2)
        self.upsample_conv.bias.data.zero_()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_embedding: torch.Tensor, condition=None) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = self.upsample_conv(x)
        x = torch.cat([x, skip], dim=1)
        for norm, res_block, attn_block, cross_attn_block in zip(self.norms, self.res_blocks, self.attn_blocks, self.cross_attn_blocks):
            # switch channel and last dimension
            x = x.permute(0, 2, 3, 1).contiguous()
            x = norm(x)
            # switch back
            x = x.permute(0, 3, 1, 2).contiguous()
            x = res_block(x, time_embedding)
            x = x + attn_block(x)
            if condition is not None and not isinstance(cross_attn_block, nn.Identity):
                x = x + cross_attn_block(x, condition)
        return x

class ConvDenoisingUNet(nn.Module):
    def __init__(
            self,
            activation,
            self_attn_class,
            cross_attn_class,
            norm_class,
            stride: Union[int, tuple[int, int]] = 1,
            in_channels: int = 3,
            model_channels: int = 64,
            out_channels: int = 3,
            channel_multipliers: list[int] = [2, 4, 8],
            time_embedding_dim: int = 256,
            attention_levels: list[bool] = [False, False, True, True],
            num_res_blocks: int = 2,
            dropout_p: float = 0.1,
            has_condition: bool = False,
            context_dim: Optional[int] = None,
            down_block_self_attn_n_heads=6,
            down_block_self_attn_d_queries=64,
            down_block_self_attn_d_values=64,
            down_block_self_attn_use_flash_attention=True,
            up_block_self_attn_n_heads=6,
            up_block_self_attn_d_queries=64,
            up_block_self_attn_d_values=64,
            up_block_self_attn_use_flash_attention=True,
            cross_attn_n_heads=6,
            cross_attn_d_queries=64,
            cross_attn_d_values=64,
            cross_attn_use_flash_attention=True,
    ):
        super().__init__()
        self.use_gradient_checkpointing = False

        activation_type = get_activation_type(activation)
        self.time_embedding = SinusoidalPositionEmbeddings(model_channels)
        self.time_transform = nn.Sequential(
            nn.Linear(model_channels, time_embedding_dim),
            activation_type() if activation_type is not activations.SwiGLU else activations.SwiGLU(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        channels = [model_channels]
        for multiplier in channel_multipliers:
            channels.append(model_channels * multiplier)

        self.down_blocks = nn.ModuleList()
        for i in range(len(channel_multipliers)):
            self.down_blocks.append(
                DownBlock(
                    stride,
                    channels[i],
                    channels[i + 1],
                    time_embedding_dim,
                    activation,
                    self_attn_class,
                    cross_attn_class,
                    norm_class,
                    has_attn=attention_levels[i],
                    has_condition=has_condition,
                    context_dim=context_dim,
                    num_res_blocks=num_res_blocks,
                    dropout_p=dropout_p,
                    self_attn_n_heads=down_block_self_attn_n_heads,
                    self_attn_d_queries=down_block_self_attn_d_queries,
                    self_attn_d_values=down_block_self_attn_d_values,
                    self_attn_use_flash_attention=down_block_self_attn_use_flash_attention,
                    cross_attn_n_heads=cross_attn_n_heads,
                    cross_attn_d_queries=cross_attn_d_queries,
                    cross_attn_d_values=cross_attn_d_values,
                    cross_attn_use_flash_attention=cross_attn_use_flash_attention,
                )
            )

        self.middle_res_block = ResidualBlock(
            channels[-1], channels[-1], time_embedding_dim, activation, norm_class
        )
        self.middle_attn_norm = norms.RMSNorm(channels[-1])

        self.middle_self_attn_block = self_attn_class(
            channels[-1], down_block_self_attn_n_heads, down_block_self_attn_d_queries, down_block_self_attn_d_values, use_flash_attention=down_block_self_attn_use_flash_attention, dropout_p=dropout_p, is_linear_attention=False
        )
        self.middle_cross_attn_block = cross_attn_class(
            channels[-1], cross_attn_n_heads, cross_attn_d_queries, cross_attn_d_values, context_dim=context_dim, use_flash_attention=cross_attn_use_flash_attention, dropout_p=dropout_p
        ) if has_condition else nn.Identity()

        self.middle_res_block2 = ResidualBlock(
            channels[-1], channels[-1], time_embedding_dim, activation, norm_class
        )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channel_multipliers))):
            self.up_blocks.append(
                UpBlock(
                    channels[i + 1],
                    channels[i],
                    time_embedding_dim,
                    activation,
                    self_attn_class,
                    cross_attn_class,
                    norm_class,
                    has_attn=attention_levels[i],
                    num_res_blocks=num_res_blocks,
                    dropout_p=dropout_p,
                    has_condition=has_condition,
                    context_dim=context_dim,
                    self_attn_n_heads=up_block_self_attn_n_heads,
                    self_attn_d_queries=up_block_self_attn_d_queries,
                    self_attn_d_values=up_block_self_attn_d_values,
                    self_attn_use_flash_attention=up_block_self_attn_use_flash_attention,
                    cross_attn_n_heads=cross_attn_n_heads,
                    cross_attn_d_queries=cross_attn_d_queries,
                    cross_attn_d_values=cross_attn_d_values,
                    cross_attn_use_flash_attention=cross_attn_use_flash_attention,
                )
            )

        self.final_res_block = ResidualBlock(
            model_channels*2, model_channels, time_embedding_dim, activation, norm_class
        )
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        # Initialize init_conv to preserve input variance
        # For 3x3 conv with C_in channels: output_std = input_std × weight_std × sqrt(9 × C_in)
        # To preserve std=1.0: weight_std = 1.0 / sqrt(9 × C_in)
        # For C_in=4 (latent channels): weight_std = 1.0 / sqrt(36) = 0.167
        # For C_in=3 (image channels): weight_std = 1.0 / sqrt(27) = 0.192
        # Using 0.15 as a reasonable value that works for both
        self.init_conv.weight.data.normal_(0.0, 0.15)
        self.init_conv.bias.data.zero_()
        # Initialize final_conv to produce output with std ≈ 1.0
        # For 3x3 conv with C_in=model_channels:
        # weight_std = 1.0 / (input_std × sqrt(9 × C_in))
        # Empirically: 0.028 gives output_std ≈ 1.22, so scale by 1.0/1.22 ≈ 0.82
        # New weight_std = 0.028 × 0.82 ≈ 0.023
        self.final_conv.weight.data.normal_(0.0, 0.023)
        self.final_conv.bias.data.zero_()

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, condition: Optional[torch.Tensor]=None) -> torch.Tensor:
        time_embedding: torch.Tensor = self.time_embedding(timesteps)
        time_embedding = self.time_transform(time_embedding.to(x.dtype))

        assert len(x.shape) == 4, f"expected 4 dimensions (batch, channel, height, width), got {len(x.shape)}"

        h: torch.Tensor = self.init_conv(x)
        initial_h = h

        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_embedding, condition=condition)
            skips.append(skip)

        if self.use_gradient_checkpointing and self.training:
            h = checkpoint.checkpoint(self.middle_res_block, h, time_embedding)
        else:
            h = self.middle_res_block(h, time_embedding)

        h = h.permute(0, 2, 3, 1).contiguous()
        h = self.middle_attn_norm(h)
        h = h.permute(0, 3, 1, 2).contiguous()
        h = h + self.middle_self_attn_block(h)
        if condition is not None:
            h = h + self.middle_cross_attn_block(h, condition)
        h = self.middle_res_block2(h, time_embedding)

        for i, (up_block, skip) in enumerate(zip(self.up_blocks, reversed(skips))):
            h = up_block(h, skip, time_embedding, condition=condition)

        h = torch.cat([h, initial_h], dim=1)
        h = self.final_res_block(h, time_embedding)
        h = self.final_conv(h)

        return h


class DiTModulation(nn.Module):
    """
    Modulation layer for DiT that produces scale and shift parameters from conditioning.
    Used for adaLN-Zero: applies learned zero-initialized gating for stable training.

    Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
    """
    def __init__(self, hidden_size: int, n_modulations: int = 6, activation_name: str = 'silu'):
        super().__init__()
        self.n_modulations = n_modulations
        self.silu = get_activation_type(activation_name)()
        self.linear = nn.Linear(hidden_size, n_modulations * hidden_size)

        self._init_weights()

    def _init_weights(self):
        # Zero-initialize the output projection for stable training
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            c: Conditioning embedding [B, D]
        Returns:
            Tuple of n_modulations tensors, each [B, 1, D] for broadcasting
        """
        c = self.silu(c)
        modulations: torch.Tensor = self.linear(c)  # [B, n_modulations * D]
        modulations = modulations.chunk(self.n_modulations, dim=-1)
        # Add sequence dimension for broadcasting: [B, D] -> [B, 1, D]
        return tuple(m.unsqueeze(1) for m in modulations)


class DiTBlock(nn.Module):
    """
    DiT Transformer block with adaLN-Zero conditioning and optional cross-attention.

    Architecture:
        1. Pre-norm with adaptive layer norm (modulated by timestep embedding)
        2. Self-attention with zero-initialized output gating
        3. Pre-norm with adaptive layer norm (if context provided)
        4. Cross-attention to context sequence with zero-initialized output gating
        5. Pre-norm with adaptive layer norm
        6. MLP with zero-initialized output gating

    The "Zero" in adaLN-Zero means the gate values are initialized to zero,
    so at initialization the block is an identity function - this provides
    stable training dynamics.

    For text-to-image generation (Stable Diffusion style):
        - Timestep conditions the block via adaLN (scale/shift/gate)
        - Text conditions the block via cross-attention (patches attend to text tokens)
    """
    def __init__(
        self,
        hidden_size: int,
        n_self_attn_heads: int,
        n_cross_attn_heads: int,
        context_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_self_attn_heads = n_self_attn_heads
        self.self_attn_head_dim = hidden_size // n_self_attn_heads
        self.n_cross_attn_heads = n_cross_attn_heads
        self.cross_attn_head_dim = hidden_size // n_cross_attn_heads
        self.context_dim = context_dim
        self.use_flash_attention = use_flash_attention

        # Layer norms (will be modulated by adaLN)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self-attention
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        # Cross-attention (for text/context conditioning)
        if context_dim is not None:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.q_cross = nn.Linear(hidden_size, hidden_size, bias=True)
            self.kv_cross = nn.Linear(context_dim, 2 * hidden_size, bias=True)
            self.cross_attn_proj = nn.Linear(hidden_size, hidden_size)
            self.cross_attn_dropout = nn.Dropout(dropout)
        else:
            self.norm_cross = None
            self.q_cross = None
            self.kv_cross = None
            self.cross_attn_proj = None
            self.cross_attn_dropout = None

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        # Modulation from timestep embedding:
        # - Self-attention: shift, scale, gate (3)
        # - Cross-attention: shift, scale, gate (3) - only if context_dim is set
        # - MLP: shift, scale, gate (3)
        n_modulations = 9 if context_dim is not None else 6
        self.adaLN_modulation = DiTModulation(hidden_size, n_modulations=n_modulations)

        self._init_weights()

    def _init_weights(self):
        # Initialize self-attention
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.attn_proj.weight)
        nn.init.zeros_(self.attn_proj.bias)

        # Initialize cross-attention if present
        if self.q_cross is not None:
            nn.init.xavier_uniform_(self.q_cross.weight)
            nn.init.zeros_(self.q_cross.bias)
            nn.init.xavier_uniform_(self.kv_cross.weight)
            nn.init.zeros_(self.kv_cross.bias)
            nn.init.xavier_uniform_(self.cross_attn_proj.weight)
            nn.init.zeros_(self.cross_attn_proj.bias)

        # Initialize MLP
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention."""
        B, N, C = x.shape

        # Compute Q, K, V
        qkv: torch.Tensor = self.qkv(x)  # [B, N, 3*C]
        qkv = qkv.reshape(B, N, 3, self.n_self_attn_heads, self.self_attn_head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)  # Each: [B, H, N, D]

        # Scaled dot-product attention (uses flash attention when available)
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            # Manual attention
            scale = self.self_attn_head_dim ** -0.5
            attn: torch.Tensor = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            attn_out = attn @ v

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.attn_proj(attn_out)

        return attn_out

    def _cross_attention(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Multi-head cross-attention where x attends to context."""
        B, N, C = x.shape
        _, S, _ = context.shape  # S = context sequence length

        # Compute Q from x, K/V from context
        q: torch.Tensor = self.q_cross(x)  # [B, N, C]
        kv: torch.Tensor = self.kv_cross(context)  # [B, S, 2*C]

        q = q.reshape(B, N, self.n_cross_attn_heads, self.cross_attn_head_dim).transpose(1, 2)  # [B, H, N, D]
        kv = kv.reshape(B, S, 2, self.n_cross_attn_heads, self.cross_attn_head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, H, S, D]
        k, v = kv.unbind(0)  # Each: [B, H, S, D]

        # Scaled dot-product attention
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.cross_attn_dropout.p if self.training else 0.0,
            )
        else:
            # Manual attention
            scale = self.cross_attn_head_dim ** -0.5
            attn: torch.Tensor = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = self.cross_attn_dropout(attn)
            attn_out = attn @ v

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.cross_attn_proj(attn_out)

        return attn_out

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, N, D]
            c: Timestep embedding [B, D] (used for adaLN modulation)
            context: Optional context sequence [B, S, context_dim] for cross-attention
        Returns:
            Output tokens [B, N, D]
        """
        # Get modulation parameters from timestep embedding
        modulations = self.adaLN_modulation(c)

        if self.context_dim is not None:
            shift1, scale1, gate1, shift_cross, scale_cross, gate_cross, shift2, scale2, gate2 = modulations
        else:
            shift1, scale1, gate1, shift2, scale2, gate2 = modulations

        # Self-attention block with adaLN-Zero
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale1) + shift1  # Modulate
        attn_out = self._self_attention(x_norm)
        x = x + gate1 * attn_out  # Gated residual

        # Cross-attention block with adaLN-Zero (if context provided)
        if self.context_dim is not None and context is not None:
            x_norm = self.norm_cross(x)
            x_norm = x_norm * (1 + scale_cross) + shift_cross  # Modulate
            cross_out = self._cross_attention(x_norm, context)
            x = x + gate_cross * cross_out  # Gated residual

        # MLP block with adaLN-Zero
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale2) + shift2  # Modulate
        mlp_out = self.mlp(x_norm)
        x = x + gate2 * mlp_out  # Gated residual

        return x


class DiTFinalLayer(nn.Module):
    """
    Final layer for DiT that projects back to patch space.
    Uses adaLN for final modulation before the linear projection.
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = DiTModulation(hidden_size, n_modulations=2)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        # Zero-initialize output for stable training
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, N, D]
            c: Conditioning embedding [B, D]
        Returns:
            Output patches [B, N, patch_size^2 * out_channels]
        """
        shift, scale = self.adaLN_modulation(c)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> torch.Tensor:
    """
    Generate 2D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension (must be divisible by 2)
        grid_size: Size of the 2D grid (assumes square)
        cls_token: If True, add a position for cls token at the beginning

    Returns:
        Positional embeddings [grid_size^2, embed_dim] or [1 + grid_size^2, embed_dim]
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing='ij'), dim=0)  # [2, H, W]
    grid = grid.reshape(2, -1).T  # [H*W, 2]

    # Split embedding dimension for height and width
    half_dim = embed_dim // 2
    omega = torch.arange(half_dim // 2, dtype=torch.float32) / (half_dim // 2)
    omega = 1.0 / (10000 ** omega)  # [D/4]

    # Compute embeddings for each dimension
    pos_h = grid[:, 0:1] * omega  # [H*W, D/4]
    pos_w = grid[:, 1:2] * omega  # [H*W, D/4]

    pos_embed = torch.cat([
        torch.sin(pos_h), torch.cos(pos_h),
        torch.sin(pos_w), torch.cos(pos_w),
    ], dim=-1)  # [H*W, D]

    if cls_token:
        cls_embed = torch.zeros(1, embed_dim)
        pos_embed = torch.cat([cls_embed, pos_embed], dim=0)

    return pos_embed


class DiTBackbone(nn.Module):
    """
    Diffusion Transformer (DiT) backbone for image generation.

    This implements the architecture from "Scalable Diffusion Models with Transformers"
    (Peebles & Xie, 2023) with modern improvements:

    - Patchification: Converts 2D images/latents into patch sequences
    - 2D sine-cosine positional embeddings
    - adaLN-Zero conditioning: Adaptive layer norm with zero-initialization
    - Standard transformer blocks with GELU activation
    - Unpatchification: Converts back to 2D spatial layout

    The model conditions on timesteps via sinusoidal embeddings, and optionally
    on additional context (e.g., text embeddings) via cross-attention or addition.

    Args:
        config: MegaTransformerConfig with the following relevant fields:
            - hidden_size: Transformer hidden dimension
            - n_layers: Number of transformer blocks
            - n_heads: Number of attention heads
            - hidden_dropout_prob: Dropout probability
            - image_size: Input image/latent spatial size
            - image_encoder_patch_size: Patch size for patchification
    """
    def __init__(self, hidden_size, n_layers, n_self_attn_heads, n_cross_attn_heads, dropout, mlp_ratio, patch_size, in_channels, context_dim, image_size):
        super().__init__()
        self.use_gradient_checkpointing = False

        # Extract config values with defaults suitable for DiT
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_self_attn_heads = n_self_attn_heads
        self.n_cross_attn_heads = n_cross_attn_heads
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio

        # Patch embedding parameters
        self.patch_size = patch_size
        self.in_channels = in_channels  # Latent channels from VAE
        self.out_channels = self.in_channels  # Predict same number of channels

        self.context_dim = context_dim

        # Compute grid size (number of patches per side)
        image_size = image_size  # Latent size, not full image
        self.grid_size = image_size // self.patch_size
        self.num_patches = self.grid_size ** 2

        # Patch embedding: project patches to hidden dimension
        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Positional embeddings (2D sine-cosine, not learned)
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))  # [1, N, D]

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        self.context_proj = nn.Linear(self.context_dim, self.hidden_size)

        # Transformer blocks with optional cross-attention
        # If context_dim is set, blocks will have cross-attention layers
        cross_attn_dim = self.hidden_size if self.context_dim is not None else None
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=self.hidden_size,
                n_self_attn_heads=self.n_self_attn_heads,
                n_cross_attn_heads=self.n_cross_attn_heads,
                context_dim=cross_attn_dim,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                use_flash_attention=True,
            )
            for _ in range(self.n_layers)
        ])

        # Final layer
        self.final_layer = DiTFinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize patch embedding
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)

        # Initialize timestep embedding MLP
        for module in self.time_embed:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch sequence.

        Args:
            x: Input image [B, C, H, W]
        Returns:
            Patch sequence [B, N, D] where N = (H/P) * (W/P)
        """
        x = self.patch_embed(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch sequence back to image.

        Args:
            x: Patch sequence [B, N, patch_size^2 * C]
        Returns:
            Image [B, C, H, W]
        """
        B = x.shape[0]
        x = x.reshape(B, self.grid_size, self.grid_size, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, H/P, P, W/P, P]
        x = x.reshape(B, self.out_channels, self.grid_size * self.patch_size, self.grid_size * self.patch_size)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT backbone.

        Args:
            x: Noisy input [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            condition: Optional conditioning tensor. Can be:
                - Class labels [B] (integers) if num_classes > 0
                - Context embeddings [B, T, context_dim] if context_dim is set
                  (will be projected and used for cross-attention)
                - None for unconditional generation

        Returns:
            Predicted noise or v [B, C, H, W]
        """
        # Ensure input dtype matches model (for bf16/fp16 training with DeepSpeed)
        model_dtype = self.patch_embed.weight.dtype
        x = x.to(dtype=model_dtype)
        if condition is not None:
            condition = condition.to(dtype=model_dtype)

        # Patchify input
        x = self.patchify(x)  # [B, N, D]

        # Add positional embeddings
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Compute timestep embedding
        t_emb = self.time_embed(timesteps.to(x.dtype))  # [B, D]

        # Process conditioning
        c = t_emb  # Timestep embedding for adaLN modulation
        context = None  # Context sequence for cross-attention

        # Text-conditional: condition is [B, T, context_dim]
        # Project context sequence for cross-attention
        if condition is not None and self.context_dim is not None:
            context = self.context_proj(condition)  # [B, T, hidden_size]

            # Also add pooled context to timestep embedding for global conditioning
            pooled_context = context.mean(dim=1)  # [B, hidden_size]
            c = c + pooled_context

        # Apply transformer blocks with cross-attention
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, c, context, use_reentrant=False)
            else:
                x = block(x, c, context)

        # Final layer
        x = self.final_layer(x, c)  # [B, N, patch_size^2 * C]

        # Unpatchify to image
        x = self.unpatchify(x)  # [B, C, H, W]

        return x


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            config: configuration.MegaTransformerConfig,
            unet: nn.Module,
            in_channels: int,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            num_timesteps: int = 1000,
            betas_schedule="linear",
            min_snr_loss_weight: bool = False,
            min_snr_gamma: float = 5.0,
            normalize: bool = False,
            ddim_sampling_eta = 0.0,
            sampling_timesteps=None,
            prediction_type: str = "epsilon",  # "epsilon" or "v"
            cfg_dropout_prob: float = 0.1,  # Probability of dropping conditioning during training for CFG
            zero_terminal_snr: bool = True,  # Rescale schedule so final timestep has SNR=0
            offset_noise_strength: float = 0.1,  # Strength of offset noise for better brightness range
            timestep_sampling: str = "logit_normal",  # "uniform" or "logit_normal" (biased toward middle)
            logit_normal_mean: float = 0.0,  # Mean for logit-normal sampling (0 = centered)
            logit_normal_std: float = 1.0,  # Std for logit-normal sampling (lower = more peaked)
    ):
        super().__init__()

        self.cfg_dropout_prob = cfg_dropout_prob
        self.offset_noise_strength = offset_noise_strength
        self.timestep_sampling = timestep_sampling
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

        self.config = config

        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        self.is_normalize = normalize
        self.ddim_sampling_eta = ddim_sampling_eta

        if sampling_timesteps is None:
            self.sampling_timesteps = num_timesteps
            self.is_ddim_sampling = False
        else:
            self.sampling_timesteps = sampling_timesteps
            assert sampling_timesteps <= num_timesteps, f"Sampling timesteps {sampling_timesteps} must be less than or equal to total timesteps {num_timesteps}."
            self.is_ddim_sampling = sampling_timesteps < num_timesteps

        self.unet = unet

        if betas_schedule == "cosine":
            betas = self.cosine_beta_schedule(num_timesteps)
        elif betas_schedule == "sigmoid":
            betas = self.sigmoid_beta_schedule(num_timesteps)
        elif betas_schedule == "karras":
            betas = self.karras_beta_schedule(num_timesteps)
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps)

        self.register_buffer("betas", betas)

        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)

        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Apply zero terminal SNR rescaling if enabled
        # This ensures the final timestep has SNR=0 (pure noise), fixing train/inference mismatch
        if zero_terminal_snr:
            alphas_cumprod = self._enforce_zero_terminal_snr(alphas_cumprod)

        self.register_buffer("alphas_cumprod", alphas_cumprod)

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt((1. - alphas_cumprod) / alphas_cumprod))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        loss_weight = maybe_clipped_snr / snr

        self.register_buffer('loss_weight', loss_weight)

        # Prediction type: "epsilon" (predict noise) or "v" (predict velocity)
        self.prediction_type = prediction_type

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def sigmoid_beta_schedule(self, timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def karras_beta_schedule(self, timesteps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        """
        Karras noise schedule from the EDM paper (Karras et al. 2022).

        Creates sigmas evenly spaced in (sigma^(1/rho)) space, which gives
        smooth transitions in log-SNR space - ideal for DPM-Solver++.

        Reference: "Elucidating the Design Space of Diffusion-Based Generative Models"
        https://arxiv.org/abs/2206.00364

        Args:
            timesteps: Number of diffusion timesteps
            sigma_min: Minimum sigma (noise level at t=0), default 0.002 from EDM
            sigma_max: Maximum sigma (noise level at t=T), default 80.0 from EDM
            rho: Schedule curvature parameter, default 7.0 from EDM

        Returns:
            betas tensor of shape [timesteps]
        """
        # Create rho-spaced sigmas (evenly spaced in sigma^(1/rho) space)
        ramp = torch.linspace(0, 1, timesteps + 1)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

        # Reverse so sigma decreases with timestep (t=0 has lowest noise)
        sigmas = sigmas.flip(0)

        # Convert sigmas to alphas_cumprod: alpha = 1 / (1 + sigma^2)
        alphas_cumprod = 1.0 / (1.0 + sigmas ** 2)

        # Ensure proper boundaries
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-5, max=1.0 - 1e-5)

        # Convert alphas_cumprod to betas
        # alpha_cumprod[t] = prod(1 - beta[0:t])
        # So: alpha[t] = alpha_cumprod[t] / alpha_cumprod[t-1] = 1 - beta[t]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1.0 - alphas

        return torch.clip(betas, 1e-5, 0.999)

    def _enforce_zero_terminal_snr(self, alphas_cumprod: torch.Tensor) -> torch.Tensor:
        """
        Rescale alphas_cumprod so that the final timestep has SNR=0 (pure noise).

        Without this fix, standard schedules (cosine, linear) never reach SNR=0,
        meaning during training the model never sees pure noise, but during
        inference we start from pure noise. This mismatch causes issues with
        generating very dark or very bright images.

        Reference: "Common Diffusion Noise Schedules and Sample Steps are Flawed"
        https://arxiv.org/abs/2305.08891
        """
        # Shift so last value becomes ~0 while preserving relative spacing
        # Using sqrt rescaling to maintain signal-to-noise ratio relationships
        alphas_cumprod_sqrt = torch.sqrt(alphas_cumprod)

        # Shift so that the final value is a small epsilon (for numerical stability)
        terminal_alpha = 1e-5
        alphas_cumprod_sqrt = alphas_cumprod_sqrt - alphas_cumprod_sqrt[-1] + math.sqrt(terminal_alpha)

        # Square back and clip for safety
        alphas_cumprod = torch.clamp(alphas_cumprod_sqrt ** 2, min=terminal_alpha, max=1.0 - 1e-5)

        return alphas_cumprod

    def normalize(self, x_0):
        return x_0 * 2 - 1

    def unnormalize(self, x):
        return (x + 1) * 0.5

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        b, *_ = t.shape
        out = a.gather(-1, t.long())
        return out.reshape(b, *((1,) * (len(shape) - 1))).to(t.device)

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps for training based on the configured sampling strategy.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on

        Returns:
            Tensor of shape (batch_size,) with timesteps in [0, num_timesteps)
        """
        if self.timestep_sampling == "logit_normal":
            # Logit-normal distribution: sample from normal, apply sigmoid, scale to timesteps
            # This biases sampling toward middle timesteps where learning signal is strongest
            # Reference: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
            u = torch.randn(batch_size, device=device) * self.logit_normal_std + self.logit_normal_mean
            t = torch.sigmoid(u)  # Maps to (0, 1), concentrated around 0.5
            t = (t * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)
        else:
            # Default: uniform sampling
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        return t

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise (epsilon prediction).

        x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * noise
        x_0 = (x_t - sqrt(1-alpha) * noise) / sqrt(alpha)
        x_0 = sqrt(1/alpha) * x_t - sqrt((1-alpha)/alpha) * noise
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted v (v prediction).

        v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * x_0
        Solving for x_0:
        x_0 = sqrt(alpha_cumprod) * x_t - sqrt(1 - alpha_cumprod) * v
        """
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_noise_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Predict noise (epsilon) from x_t and predicted v.

        v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * x_0
        noise = sqrt(1 - alpha_cumprod) * x_t + sqrt(alpha_cumprod) * v
        """
        return (
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t +
            self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * v
        )

    def get_v_target(self, x_start: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute v target for training: v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * x_0"""
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def model_predictions(self, x, t, condition=None, clip_x_start=False, guidance_scale=1.0):
        """
        Get model predictions, optionally with classifier-free guidance.

        Args:
            x: Noisy input
            t: Timesteps
            condition: Conditioning (e.g., text embeddings)
            clip_x_start: Whether to clip predicted x_start to [-1, 1]
            guidance_scale: CFG scale. 1.0 = no guidance, >1.0 = stronger conditioning
        """
        # Apply classifier-free guidance if scale > 1 and we have conditioning
        if guidance_scale != 1.0 and condition is not None:
            # Run model twice: once unconditional, once conditional
            uncond_condition = torch.zeros_like(condition)

            # Unconditional prediction
            uncond_output = self.unet(x, t, uncond_condition)
            # Conditional prediction
            cond_output = self.unet(x, t, condition)

            # CFG interpolation: output = uncond + scale * (cond - uncond)
            model_output = uncond_output + guidance_scale * (cond_output - uncond_output)
        else:
            model_output = self.unet(x, t, condition)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else nn.Identity()

        if self.prediction_type == "v":
            # Model predicts v, convert to x_start and noise
            x_start = self.predict_start_from_v(x, t, model_output)
            pred_noise = self.predict_noise_from_v(x, t, model_output)
        else:
            # Model predicts noise (epsilon)
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        x_start = maybe_clip(x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, condition=None, clip_denoised=True, guidance_scale=1.0):
        _, x_start = self.model_predictions(x, t, condition=condition, guidance_scale=guidance_scale)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, condition=None, guidance_scale=1.0):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x = x, t = batched_times, condition=condition, clip_denoised=True, guidance_scale=guidance_scale
        )

        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0

        pred_x0 = model_mean + torch.exp(0.5 * model_log_variance) * noise

        return pred_x0, x_start

    @torch.no_grad()
    def p_sample_loop(self, x, condition=None, return_intermediate: bool=False, guidance_scale=1.0):
        noise_preds = []
        x_start_preds = []
        for time_step in reversed(range(0, self.num_timesteps)):
            x, x_start = self.p_sample(x, time_step, condition=condition, guidance_scale=guidance_scale)
            if return_intermediate:
                noise_preds.append(x)
                x_start_preds.append(x_start)
        return (x, noise_preds, x_start_preds) if return_intermediate else x

    @torch.no_grad()
    def ddim_sample_loop(self, x, condition=None, return_intermediate=False, override_sampling_steps=None, guidance_scale=1.0):
        sampling_timesteps = override_sampling_steps if override_sampling_steps is not None else self.sampling_timesteps

        times = torch.linspace(-1, self.num_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn_like(x, device=x.device)

        noise_preds = []
        x_start_preds = []

        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((x.shape[0],), time, device=x.device, dtype = torch.long)
            pred_noise, x_start = self.model_predictions(img, time_cond, condition=condition, clip_x_start=False, guidance_scale=guidance_scale)

            # Clamp x_start to prevent sampling blowup
            # For latent diffusion, latents typically range [-6, 6], so [-10, 10] is safe
            x_start = torch.clamp(x_start, min=-10.0, max=10.0)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            # Clamp denominators to prevent division by zero at low timesteps (cosine schedule)
            one_minus_alpha = (1 - alpha).clamp(min=1e-8)
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / one_minus_alpha).sqrt()
            c = (1 - alpha_next - sigma ** 2).clamp(min=0).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if return_intermediate:
                noise_preds.append(img)
                x_start_preds.append(x_start)

        return (img, noise_preds, x_start_preds) if return_intermediate else img

    def _get_sigmas_from_alphas_cumprod(self) -> torch.Tensor:
        """Convert alphas_cumprod to sigmas for DPM-Solver."""
        return torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)

    def _get_lambdas_from_alphas_cumprod(self) -> torch.Tensor:
        """Get log-SNR (lambda) values from alphas_cumprod."""
        # lambda = log(alpha / sigma) = log(alpha / sqrt(1 - alpha^2))
        # = 0.5 * log(alpha^2 / (1 - alpha^2)) = 0.5 * log(alpha_cumprod / (1 - alpha_cumprod))
        return 0.5 * torch.log(self.alphas_cumprod / (1 - self.alphas_cumprod))

    @torch.no_grad()
    def dpm_solver_pp_sample_loop(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor]=None,
        return_intermediate=False,
        override_sampling_steps=None,
        guidance_scale=1.0,
        order=2,  # 1, 2, or 3
    ):
        """
        DPM-Solver++ sampler for fast high-quality sampling.

        Reference: "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
        https://arxiv.org/abs/2211.01095

        Args:
            x: Initial noise tensor (used for shape/device reference)
            condition: Conditioning tensor
            return_intermediate: Whether to return intermediate steps
            override_sampling_steps: Number of sampling steps (default: self.sampling_timesteps)
            guidance_scale: CFG guidance scale
            order: Solver order (1=Euler, 2=second-order, 3=third-order). 2 is recommended.

        Returns:
            Denoised sample, optionally with intermediate predictions
        """
        sampling_timesteps = override_sampling_steps if override_sampling_steps is not None else self.sampling_timesteps
        device = x.device

        # Create timestep schedule (uniformly spaced in timestep space)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, sampling_timesteps + 1, device=device)
        timesteps = timesteps.long()

        # Get alpha and sigma values for the schedule (standard parameterization)
        # alpha_t = sqrt(alpha_cumprod), sigma_t = sqrt(1 - alpha_cumprod)
        alphas = torch.sqrt(self.alphas_cumprod[timesteps])
        sigmas = torch.sqrt(1 - self.alphas_cumprod[timesteps])

        # Clamp both alphas and sigmas to avoid numerical issues
        # When alpha -> 0 (at t=T with ZTSNR), log(alpha/sigma) -> -inf which breaks lambda
        # When sigma -> 0 (at t=0), log(alpha/sigma) -> inf which also breaks lambda
        alphas = alphas.clamp(min=1e-5)
        sigmas = sigmas.clamp(min=1e-5)

        # Initialize with noise
        img = torch.randn_like(x, device=device)

        noise_preds = []
        x_start_preds = []

        # For multistep methods, store previous model outputs (x_0 predictions)
        model_outputs = []

        for i in range(sampling_timesteps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            # Get current alpha/sigma
            alpha_curr = alphas[i]
            alpha_next = alphas[i + 1]
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Lambda (log-SNR) values: lambda = log(alpha/sigma)
            lambda_curr = torch.log(alpha_curr / sigma_curr)
            lambda_next = torch.log(alpha_next / sigma_next)
            h = lambda_next - lambda_curr  # Step size in lambda space

            # Get model prediction at current timestep
            t_cond = torch.full((x.shape[0],), t_curr.item(), device=device, dtype=torch.long)
            _, x_start = self.model_predictions(
                img, t_cond, condition=condition, clip_x_start=False, guidance_scale=guidance_scale
            )

            # Clamp x_start to prevent sampling blowup
            # For latent diffusion, latents typically range [-6, 6], so [-10, 10] is safe
            x_start = torch.clamp(x_start, min=-10.0, max=10.0)

            # Store x_0 prediction for multistep
            model_outputs.append(x_start)
            if len(model_outputs) > order:
                model_outputs.pop(0)

            if return_intermediate:
                noise_preds.append(img.clone())
                x_start_preds.append(x_start)

            # DPM-Solver++ update (data prediction formulation)
            ratio = sigma_next / sigma_curr

            # Handle degenerate case: when ratio ≈ 1.0, the update formula becomes img = img (no update)
            # This happens with cosine schedule when sigma is clamped at low timesteps
            # In this case, just use x_start directly since we're effectively at t=0
            # Use strict threshold (0.9999) to only catch truly degenerate cases, not normal high-noise steps
            if ratio > 0.9999:
                img = x_start
            elif order == 1 or len(model_outputs) == 1:
                # First-order update: x_{t-1} = (sigma_{t-1}/sigma_t) * x_t + alpha_{t-1} * (1 - sigma_{t-1}/sigma_t) * x_0
                # Simplified using exponential: ratio = sigma_next/sigma_curr = exp(lambda_curr - lambda_next) = exp(-h)
                img = ratio * img + alpha_next * (1 - ratio) * x_start

            elif order == 2 and len(model_outputs) >= 2:
                # Second-order multistep (DPM-Solver++-2M)
                x_start_prev = model_outputs[-2]

                # Get previous lambda for ratio calculation
                if i > 0:
                    alpha_prev = alphas[i - 1]
                    sigma_prev = sigmas[i - 1]
                    lambda_prev = torch.log(alpha_prev / sigma_prev)
                    h_prev = lambda_curr - lambda_prev
                    r = h_prev / h if h != 0 else 1.0
                else:
                    r = 1.0

                # Second-order correction term
                D0 = x_start
                D1 = (1 / (2 * r)) * (x_start - x_start_prev) if r != 0 else torch.zeros_like(x_start)

                # Update
                img = ratio * img + alpha_next * (1 - ratio) * D0 + alpha_next * (1 - ratio) * D1

            elif order == 3 and len(model_outputs) >= 3:
                # Third-order - fall back to second order for simplicity
                # Full third-order is complex and rarely needed
                x_start_prev = model_outputs[-2]

                if i > 0:
                    alpha_prev = alphas[i - 1]
                    sigma_prev = sigmas[i - 1]
                    lambda_prev = torch.log(alpha_prev / sigma_prev)
                    h_prev = lambda_curr - lambda_prev
                    r = h_prev / h if h != 0 else 1.0
                else:
                    r = 1.0

                D0 = x_start
                D1 = (1 / (2 * r)) * (x_start - x_start_prev) if r != 0 else torch.zeros_like(x_start)

                img = ratio * img + alpha_next * (1 - ratio) * D0 + alpha_next * (1 - ratio) * D1

            else:
                # Fallback to first-order
                img = ratio * img + alpha_next * (1 - ratio) * x_start

        # img should now be close to x_0
        return (img, noise_preds, x_start_preds) if return_intermediate else img

    @torch.no_grad()
    def sample(self, device, batch_size: int, condition: Optional[torch.Tensor]=None, return_intermediate: bool=False, override_sampling_steps: Optional[int]=None, generator=None, guidance_scale: float=1.0, sampler: str="dpm_solver_pp", dpm_solver_order: int=2, **kwargs) -> torch.Tensor:
        """
        Sample from the diffusion model.

        Args:
            device: Device to run on
            batch_size: Number of samples to generate
            condition: Conditioning tensor (e.g., text embeddings)
            return_intermediate: If True, return intermediate denoising steps
            override_sampling_steps: Override default sampling steps
            generator: Optional random generator for reproducibility
            guidance_scale: Classifier-free guidance scale. 1.0 = no guidance, >1.0 = stronger conditioning.
                           Typical values are 3.0-15.0 for text-to-image.
            sampler: Sampling algorithm. Options:
                - "dpm_solver_pp": DPM-Solver++ (recommended, fast and high quality)
                - "ddim": DDIM sampler
                - "ddpm": Full DDPM sampler (slow, uses all timesteps)
            dpm_solver_order: Order for DPM-Solver++ (1, 2, or 3). 2 is recommended.
        """
        image_size = kwargs.get('image_size', self.config.image_size)

        x = torch.randn(batch_size, self.in_channels, image_size, image_size, device=device, generator=generator)

        sampling_steps = override_sampling_steps if override_sampling_steps is not None else self.sampling_timesteps

        if sampler == "dpm_solver_pp":
            x = self.dpm_solver_pp_sample_loop(
                x, condition=condition, return_intermediate=return_intermediate,
                override_sampling_steps=sampling_steps, guidance_scale=guidance_scale,
                order=dpm_solver_order
            )
        elif sampler == "ddim" or (self.is_ddim_sampling and sampler != "ddpm"):
            x = self.ddim_sample_loop(x, condition=condition, return_intermediate=return_intermediate, override_sampling_steps=sampling_steps, guidance_scale=guidance_scale)
        else:
            # ddpm - full sampling
            x = self.p_sample_loop(x, condition=condition, return_intermediate=return_intermediate, guidance_scale=guidance_scale)

        if return_intermediate:
            img, noise_preds, x_start_preds = x
        else:
            img = x

        if self.is_normalize:
            img = self.unnormalize(img)
            if return_intermediate:
                # Also unnormalize intermediate x_start predictions
                x_start_preds = [self.unnormalize(x_start) for x_start in x_start_preds]

        return (img, noise_preds, x_start_preds) if return_intermediate else img
    
    @autocast('cuda', enabled = False)
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor]=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise=None, condition=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # Offset noise: add constant noise across spatial dimensions to help model
        # learn to generate images with varying overall brightness/darkness
        # Reference: https://www.crosslabs.org/blog/diffusion-with-offset-noise
        if self.offset_noise_strength > 0 and self.training:
            # noise shape is [B, C, H, W] for images or [B, C, H, W] for audio
            # We add noise that's constant across H, W dimensions
            offset_noise = torch.randn(x_start.shape[0], x_start.shape[1], 1, 1, device=x_start.device, dtype=x_start.dtype)
            noise = noise + self.offset_noise_strength * offset_noise

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # CFG: randomly drop conditioning during training
        if condition is not None and self.cfg_dropout_prob > 0 and self.training:
            # Create mask for which samples in batch should have conditioning dropped
            batch_size = x_start.shape[0]
            dropout_mask = torch.rand(batch_size, device=x_start.device) < self.cfg_dropout_prob
            # Zero out conditioning for selected samples
            # condition shape is typically [B, T, D] for text embeddings
            condition = condition.clone()
            condition[dropout_mask] = 0.0

        model_output = self.unet(x_noisy, t, condition)

        if self.prediction_type == "v":
            # Model predicts v, target is v
            target = self.get_v_target(x_start, noise, t)
        else:
            # Model predicts noise (epsilon)
            target = noise

        loss = F.mse_loss(model_output, target, reduction='none')
        loss_per_sample = loss.mean(dim=[1, 2, 3])

        loss_weights = self._extract(self.loss_weight, t, loss_per_sample.shape)

        loss_per_sample = loss_per_sample * loss_weights

        final_loss = loss_per_sample.mean()

        return model_output, final_loss

    def forward(self, x_0: torch.Tensor, condition: Optional[torch.Tensor]=None):
        """
        default impl is image diffusion
        returns model output noises and noise reconstruction loss from `p_losses` function
        """
        if len(x_0.shape) == 5:
            # squish batch and example dimensions if necessary
            b, e, c, h, w = x_0.shape
            x_0 = x_0.view(-1, c, h, w)
        else:
            b, c, h, w = x_0.shape
            e = None

        if condition is not None:
            if len(condition.shape) == 5:
                # squish batch and example dimensions if necessary
                *_, c, h, w = condition.shape
                condition = condition.view(-1, c, h, w)

        t = self._sample_timesteps(x_0.shape[0], x_0.device)

        if self.is_normalize:
            x_0 = self.normalize(x_0)

        model_output, mse_loss = self.p_losses(x_0, t, condition=condition)

        if e is not None:
            # restore model example dimension
            model_output = model_output.view(b, e, c, h, w)
        return model_output, [mse_loss]


# =============================================================================
# Flow Matching (Rectified Flow)
# =============================================================================

class FlowMatching(nn.Module):
    """
    Flow Matching / Rectified Flow implementation.

    A simpler alternative to diffusion models that uses optimal transport paths
    between noise and data distributions.

    Key differences from GaussianDiffusion:
    - Linear interpolation: x_t = (1-t)*x_0 + t*noise (no complex noise schedule)
    - Velocity prediction: model predicts v = noise - x_0
    - ODE sampling: deterministic integration from t=1 to t=0
    - Simpler training objective, often faster convergence

    Reference:
    - "Flow Matching for Generative Modeling" (Lipman et al. 2022)
    - "Rectified Flow" (Liu et al. 2022)

    Usage:
        model = FlowMatching(
            config=config,
            unet=DiTBackbone(config),
            in_channels=4,
        )

        # Training
        output, loss = model(latents, condition=text_embeddings)

        # Sampling
        samples = model.sample(device, batch_size=4, condition=text_embeddings)
    """

    def __init__(
        self,
        config: configuration.MegaTransformerConfig,
        unet: nn.Module,
        in_channels: int,
        sigma_min: float = 0.0,  # Minimum noise level (0 = pure data at t=0)
        cfg_dropout_prob: float = 0.1,
        timestep_sampling: str = "logit_normal",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ):
        """
        Args:
            config: Model configuration
            unet: Denoising backbone (DiTBackbone, ConvDenoisingUNet, etc.)
            in_channels: Number of input channels (latent channels from VAE)
            sigma_min: Minimum noise level at t=0 (usually 0)
            cfg_dropout_prob: Probability of dropping conditioning for CFG
            timestep_sampling: How to sample timesteps ("uniform" or "logit_normal")
            logit_normal_mean: Mean for logit-normal sampling
            logit_normal_std: Std for logit-normal sampling
        """
        super().__init__()

        self.config = config
        self.unet = unet
        self.in_channels = in_channels
        self.sigma_min = sigma_min
        self.cfg_dropout_prob = cfg_dropout_prob
        self.timestep_sampling = timestep_sampling
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps t ∈ [0, 1] for training.

        Args:
            batch_size: Number of samples
            device: Device to create tensor on

        Returns:
            Tensor of shape [batch_size] with values in [0, 1]
        """
        if self.timestep_sampling == "logit_normal":
            # Logit-normal distribution biased toward middle timesteps
            u = torch.randn(batch_size, device=device)
            u = self.logit_normal_mean + self.logit_normal_std * u
            t = torch.sigmoid(u)
        else:
            # Uniform sampling
            t = torch.rand(batch_size, device=device)

        # Clamp to avoid numerical issues at boundaries
        t = t.clamp(min=1e-5, max=1.0 - 1e-5)

        return t

    def interpolate(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear interpolation between data and noise.

        x_t = (1 - t) * x_0 + t * noise

        Args:
            x_0: Clean data [B, C, H, W]
            noise: Gaussian noise [B, C, H, W]
            t: Timesteps [B] in range [0, 1]

        Returns:
            Interpolated samples x_t [B, C, H, W]
        """
        t = t.view(-1, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting
        x_t = (1 - t) * x_0 + t * noise
        return x_t

    def get_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target velocity for flow matching.

        v = noise - x_0 (direction from data to noise)

        The model learns to predict this velocity, which defines the flow field.

        Args:
            x_0: Clean data [B, C, H, W]
            noise: Gaussian noise [B, C, H, W]

        Returns:
            Target velocity [B, C, H, W]
        """
        return noise - x_0

    def predict_x0_from_velocity(
        self,
        x_t: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from current state and velocity.

        Given: x_t = (1-t)*x_0 + t*noise, v = noise - x_0
        Solve: x_0 = x_t - t*v

        Args:
            x_t: Current state [B, C, H, W]
            v: Predicted velocity [B, C, H, W]
            t: Timesteps [B]

        Returns:
            Predicted x_0 [B, C, H, W]
        """
        t = t.view(-1, 1, 1, 1)
        x_0 = x_t - t * v
        return x_0

    def training_step(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute training loss for flow matching.

        Args:
            x_0: Clean data [B, C, H, W]
            condition: Optional conditioning [B, T, D]

        Returns:
            Tuple of (model_output, loss)
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample timesteps and noise
        t = self._sample_timesteps(batch_size, device)
        noise = torch.randn_like(x_0).to(x_0.dtype)

        # Interpolate to get x_t
        x_t = self.interpolate(x_0, noise, t)

        # Get target velocity
        velocity_target = self.get_velocity(x_0, noise)

        # Apply CFG dropout (drop conditioning with some probability)
        if condition is not None and self.cfg_dropout_prob > 0 and self.training:
            dropout_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob
            condition = condition.clone()
            condition[dropout_mask] = 0.0

        # Model prediction
        # Note: t needs to be passed in a format the model expects
        # For DiT, we typically pass t as floats in [0, 1]
        velocity_pred = self.unet(x_t, t, condition)

        # MSE loss on velocity
        loss = F.mse_loss(velocity_pred, velocity_target)

        return velocity_pred, loss

    @torch.no_grad()
    def euler_step(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Single Euler step for ODE integration.

        dx/dt = v(x, t) => x_{t+dt} = x_t + dt * v(x_t, t)

        For flow matching we integrate from t=1 (noise) to t=0 (data),
        so dt is negative.

        Args:
            x: Current state [B, C, H, W]
            t_curr: Current timestep
            t_next: Next timestep
            condition: Optional conditioning
            guidance_scale: CFG scale (1.0 = no guidance)

        Returns:
            Updated state [B, C, H, W]
        """
        batch_size = x.shape[0]
        device = x.device

        # Time tensor
        t = torch.full((batch_size,), t_curr, device=device, dtype=x.dtype)

        # Predict velocity
        if guidance_scale != 1.0 and condition is not None:
            # Classifier-free guidance
            v_cond = self.unet(x, t, condition)
            v_uncond = self.unet(x, t, torch.zeros_like(condition))
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = self.unet(x, t, condition)

        # Euler step (note: t_next < t_curr when going from noise to data)
        dt = t_next - t_curr
        x_next = x + dt * v

        return x_next

    @torch.no_grad()
    def heun_step(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Heun's method (improved Euler) for more accurate ODE integration.

        This is a second-order method that uses a predictor-corrector approach.

        Args:
            x: Current state [B, C, H, W]
            t_curr: Current timestep
            t_next: Next timestep
            condition: Optional conditioning
            guidance_scale: CFG scale

        Returns:
            Updated state [B, C, H, W]
        """
        batch_size = x.shape[0]
        device = x.device
        dt = t_next - t_curr

        def get_velocity(x_in, t_val):
            t = torch.full((batch_size,), t_val, device=device, dtype=x.dtype)
            if guidance_scale != 1.0 and condition is not None:
                v_cond = self.unet(x_in, t, condition)
                v_uncond = self.unet(x_in, t, torch.zeros_like(condition))
                return v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                return self.unet(x_in, t, condition)

        # Predictor (Euler step)
        v1 = get_velocity(x, t_curr)
        x_pred = x + dt * v1

        # Corrector
        v2 = get_velocity(x_pred, t_next)
        x_next = x + 0.5 * dt * (v1 + v2)

        return x_next

    @torch.no_grad()
    def sample(
        self,
        device: torch.device,
        batch_size: int,
        condition: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        solver: str = "euler",  # "euler" or "heun"
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list]]:
        """
        Generate samples using ODE integration from noise to data.

        Args:
            device: Device to generate on
            batch_size: Number of samples to generate
            condition: Optional conditioning [B, T, D]
            num_steps: Number of integration steps
            guidance_scale: Classifier-free guidance scale
            solver: ODE solver ("euler" or "heun")
            generator: Optional random generator for reproducibility
            return_intermediate: Return intermediate states

        Returns:
            Generated samples [B, C, H, W], optionally with intermediates
        """
        # Get image size from config or kwargs
        image_size = kwargs.get('image_size', getattr(self.config, 'image_size', 32))

        # Start from pure noise at t=1
        if generator is not None:
            x = torch.randn(
                batch_size, self.in_channels, image_size, image_size,
                device=device, generator=generator
            )
        else:
            x = torch.randn(
                batch_size, self.in_channels, image_size, image_size,
                device=device
            )

        # Time steps from t=1 to t=0
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        intermediates = [x.clone()] if return_intermediate else None

        # Choose solver
        step_fn = self.heun_step if solver == "heun" else self.euler_step

        # Integration loop
        for i in range(num_steps):
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()

            x = step_fn(x, t_curr, t_next, condition, guidance_scale)

            if return_intermediate:
                intermediates.append(x.clone())

        if return_intermediate:
            return x, intermediates, None
        return x

    def forward(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for training.

        Args:
            x_0: Clean data [B, C, H, W] or [B, E, C, H, W]
            condition: Optional conditioning

        Returns:
            Tuple of (model_output, [loss])
        """
        # Handle batched examples dimension
        if len(x_0.shape) == 5:
            b, e, c, h, w = x_0.shape
            x_0 = x_0.view(-1, c, h, w)
        else:
            b, c, h, w = x_0.shape
            e = None

        if condition is not None and len(condition.shape) == 4:
            condition = condition.view(-1, *condition.shape[2:])

        model_output, loss = self.training_step(x_0, condition)

        if e is not None:
            model_output = model_output.view(b, e, c, h, w)

        return model_output, loss, None


def create_flow_matching_model(
    config: configuration.MegaTransformerConfig,
    unet: nn.Module,
    latent_channels: int = 4,
    cfg_dropout_prob: float = 0.1,
    timestep_sampling: str = "logit_normal",
) -> FlowMatching:
    """
    Create a Flow Matching model with DiT backbone.

    Args:
        config: Model configuration
        latent_channels: Number of VAE latent channels
        cfg_dropout_prob: CFG dropout probability
        timestep_sampling: Timestep sampling strategy

    Returns:
        FlowMatching model
    """
    return FlowMatching(
        config=config,
        unet=unet,
        in_channels=latent_channels,
        cfg_dropout_prob=cfg_dropout_prob,
        timestep_sampling=timestep_sampling,
    )
