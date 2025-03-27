from typing import Optional, Union

from model import megatransformer_modules

import math
import megatransformer_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioDiffusionSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, use_flash_attention=True, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.use_flash_attention = use_flash_attention
        
        self.q_proj = nn.Linear(hidden_size, d_queries * n_heads)
        self.k_proj = nn.Linear(hidden_size, d_queries * n_heads)
        self.v_proj = nn.Linear(hidden_size, d_values * n_heads)
        
        self.out_proj = nn.Linear(self.d_values * n_heads, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normal multi-head self attention, but it expects 4D input where it will batch by the first and third dimensions,
        and outputs the same shape.
        Args:
            x: [B, H, W, T] where B is batch size, H is height, W is width and T is time.
        Returns:
            output: [B, H, W, T] where B is batch size, H is height, W is width and T is time. Attention is applied
            along the T dimension, between the W dimension values, batched along B*W.
        """
        B, H, W, T = x.shape

        x = x.permute(0, 2, 1, 3)  # [B, W, H, T]

        x = x.contiguous().view(-1, H, T)  # [B*W, H, T]

        x = x.permute(0, 2, 1)  # [B*W, T, H]
        
        q: torch.Tensor = self.q_proj(x)  # [B*W, T, n_heads*d_queries]
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)
        
        q = q.view(-1, T, self.n_heads, self.d_queries)  # [B*W, T, n_heads, d_queries]
        k = k.view(-1, T, self.n_heads, self.d_queries)
        v = v.view(-1, T, self.n_heads, self.d_values)
        
        q = q.transpose(1, 2)  # [B*W, n_heads, T, d_queries]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )  # [B*W, n_heads, T, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*W, n_heads, T, T]
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)  # [B*W, n_heads, T, d_queries]
        
        output = output.transpose(1, 2).contiguous()  # [B*W, T, n_heads, d_queries]

        output = output.view(-1, T, self.n_heads*self.d_values)  # [B*W, T, H]
        
        output = self.out_proj(output)  # [B*W, T, H]

        output = output.permute(0, 2, 1)  # [B*W, H, T]

        # restore input shape by splitting the hidden dim into width and height
        output = output.view(B, W, H, T)

        output = output.permute(0, 2, 1, 3)  # [B, H, W, T]
        
        return output

class AudioDiffusionCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, context_dim=None, use_flash_attention=True, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.context_dim = context_dim or hidden_size  # If None, use hidden_dim
        self.use_flash_attention = use_flash_attention
        
        self.q_proj = nn.Linear(hidden_size, n_heads*d_queries)
        self.k_proj = nn.Linear(self.context_dim, n_heads*d_queries)
        self.v_proj = nn.Linear(self.context_dim, n_heads*d_values)
        
        self.out_proj = nn.Linear(n_heads*d_values, hidden_size)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, H, W, T = x.size()
        BC, N, CH = context.size()

        assert B == BC, f"Batch size mismatch: {B} vs {BC}. Shapes: {x.shape}, {context.shape}"

        x = x.permute(0, 2, 1, 3)  # [B, W, H, T]
        x = x.contiguous().view(B*W, H, T)    # [B*W, H, T]
        x = x.permute(0, 2, 1)  # [B*W, T, H]

        # context is 3D batched linear feature tokens, broadcast along the width dimension for attention
        context = context.unsqueeze(2).expand(-1, -1, W, -1)  # [B, N, W, CH]
        context = context.permute(0, 2, 3, 1)  # [B, W, CH, N]
        context = context.contiguous().view(B*W, CH, N)   # [B*W, CH, N]
        context = context.permute(0, 2, 1)  # [B*W, N, CH]

        q: torch.Tensor = self.q_proj(x)        # [B*W, T, n_heads*d_queries]
        k: torch.Tensor = self.k_proj(context)  # [B*W, N, n_heads*d_queries]
        v: torch.Tensor = self.v_proj(context)  # [B*W, N, n_heads*d_values]

        q = q.view(-1, T, self.n_heads, self.d_queries).transpose(1, 2)  # [B*W, n_heads, T, d_queries]
        k = k.view(-1, N, self.n_heads, self.d_queries).transpose(1, 2)  # [B*W, n_heads, N, d_queries]
        v = v.view(-1, N, self.n_heads, self.d_values).transpose(1, 2)  # [B*W, n_heads, N, d_values]

        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )  # [B*W, n_heads, T, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*W, n_heads, T, N]
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)  # [B*W, n_heads, T, d_values]
        
        output = output.transpose(1, 2).contiguous()  # [B*W, T, n_heads, head_dim]
        output = output.view(-1, T, self.n_heads*self.d_values)  # [B*W, T, n_heads*d_values]
        
        output = self.out_proj(output)  # [B*W, T, H]

        output = output.permute(0, 2, 1)  # [B*W, H, T]

        # restore input shape by splitting the hidden dim into width and height
        output = output.view(B, W, H, T)

        output = output.permute(0, 2, 1, 3)  # [B, H, W, T]

        return output

class ImageDiffusionSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, use_flash_attention=True, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.use_flash_attention = use_flash_attention
        self.dropout_p = dropout  # Store dropout probability for flash attention
        
        self.q_proj = nn.Linear(hidden_size, d_queries * n_heads)
        self.k_proj = nn.Linear(hidden_size, d_queries * n_heads)
        self.v_proj = nn.Linear(hidden_size, d_values * n_heads)
        
        self.out_proj = nn.Linear(self.d_values * n_heads, hidden_size)
        
        # Always create dropout layer (needed for non-flash path)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-head self attention for image data, treating spatial dimensions (H,W) as the sequence length.
        Args:
            x: [B, C, H, W] where B is batch size, C is channels, H is height, W is width.
        Returns:
            output: [B, C, H, W] where attention is applied along the spatial dimensions.
        """
        B, C, H, W = x.shape
        seq_len = H * W

        x = x.contiguous().view(B, C, seq_len)  # [B, C, H*W]
        x = x.permute(0, 2, 1)  # [B, H*W, C]
        
        q: torch.Tensor = self.q_proj(x)  # [B, H*W, n_heads*d_queries]
        k: torch.Tensor = self.k_proj(x)  # [B, H*W, n_heads*d_queries]
        v: torch.Tensor = self.v_proj(x)  # [B, H*W, n_heads*d_values]
        
        q = q.view(B, seq_len, self.n_heads, self.d_queries)  # [B, H*W, n_heads, d_queries]
        k = k.view(B, seq_len, self.n_heads, self.d_queries)  # [B, H*W, n_heads, d_queries]
        v = v.view(B, seq_len, self.n_heads, self.d_values)  # [B, H*W, n_heads, d_values]
        
        q = q.transpose(1, 2)  # [B, n_heads, H*W, d_queries]
        k = k.transpose(1, 2)  # [B, n_heads, H*W, d_queries]
        v = v.transpose(1, 2)  # [B, n_heads, H*W, d_values]
        
        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, H*W, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, H*W, H*W]
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)  # [B, n_heads, H*W, d_values]
        
        output = output.transpose(1, 2).contiguous()  # [B, H*W, n_heads, d_values]
        output = output.view(B, seq_len, self.n_heads*self.d_values)  # [B, H*W, n_heads*d_values]
        
        output = self.out_proj(output)  # [B, H*W, C]

        # Reshape back to original spatial dimensions
        output = output.view(B, H, W, C)
        output = output.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return output

class ImageDiffusionCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, context_dim=None, use_flash_attention=True, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.context_dim = context_dim or hidden_size  # If None, use hidden_dim
        self.use_flash_attention = use_flash_attention
        
        self.q_proj = nn.Linear(hidden_size, n_heads*d_queries)
        self.k_proj = nn.Linear(self.context_dim, n_heads*d_queries)
        self.v_proj = nn.Linear(self.context_dim, n_heads*d_values)
        
        self.out_proj = nn.Linear(n_heads*d_values, hidden_size)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        tgt_seq_len = H * W
        BC, CC, HC, WC = context.size()
        ctxt_seq_len = HC * WC

        assert B == BC, f"Batch size mismatch: {B} vs {BC}"

        x = x.contiguous().view(B, C, tgt_seq_len)    # [B, C, tgt_seq_len]
        x = x.permute(0, 2, 1)  # [B, tgt_seq_len, C]

        context = context.contiguous().view(B, CC, ctxt_seq_len)   # [B, CC, ctxt_seq_len]
        context = context.permute(0, 2, 1)  # [B, ctxt_seq_len, CC]
        
        q: torch.Tensor = self.q_proj(x)        # [B, tgt_seq_len, n_heads*d_queries]
        k: torch.Tensor = self.k_proj(context)  # [B, ctxt_seq_len, n_heads*d_queries]
        v: torch.Tensor = self.v_proj(context)  # [B, ctxt_seq_len, n_heads*d_values]
        
        q = q.view(B, tgt_seq_len, self.n_heads, self.d_queries).transpose(1, 2)   # [B, n_heads, tgt_seq_len, d_queries]
        k = k.view(B, ctxt_seq_len, self.n_heads, self.d_queries).transpose(1, 2)  # [B, n_heads, ctxt_seq_len, d_queries]
        v = v.view(B, ctxt_seq_len, self.n_heads, self.d_values).transpose(1, 2)   # [B, n_heads, ctxt_seq_len, d_values]
        
        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, tgt_seq_len, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, tgt_seq_len, ctxt_seq_len]
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)  # [B, n_heads, tgt_seq_len, d_values]
        
        output = output.transpose(1, 2).contiguous()  # [B, tgt_seq_len, n_heads, d_values]
        output = output.view(B, tgt_seq_len, self.n_heads*self.d_values)  # [B, tgt_seq_len, n_heads*d_values]
        
        output = self.out_proj(output)  # [B, tgt_seq_len, C]

        output = output.permute(0, 2, 1)  # [B, C, tgt_seq_len]

        # restore input shape by splitting the hidden dim into width and height
        output = output.view(B, C, H, W)

        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, activation, dropout):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim

        activation_type = megatransformer_utils.get_activation_type(activation)
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        if time_embedding_dim is not None:
            self.time_mlp = nn.Sequential(
                activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(time_embedding_dim),
                nn.Linear(time_embedding_dim, out_channels),
            )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(out_channels),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor]=None) -> torch.Tensor:
        h = self.conv1(x)

        if self.time_embedding_dim is not None and time_embedding is not None:
            time_embedding = self.time_mlp(time_embedding)
            h = h + time_embedding.unsqueeze(-1).unsqueeze(-1)

        h = self.conv2(h)
        h = h + self.shortcut(x)
        return h

class DownBlock(nn.Module):
    def __init__(self,
                 stride,
                 in_channels: int,
                 out_channels: int,
                 time_embedding_dim: int,
                 activation,
                 self_attn_class,
                 has_attn: bool=False,
                 num_res_blocks: int=2,
                 dropout: float=0.1,
                 self_attn_n_heads=6,
                 self_attn_d_queries=64,
                 self_attn_d_values=64,
                 self_attn_use_flash_attention=True):
        super().__init__()

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, time_embedding_dim, activation, dropout)
            for i in range(num_res_blocks)
        ])

        self.attn_blocks = nn.ModuleList([
            self_attn_class(
                out_channels, self_attn_n_heads, self_attn_d_queries, self_attn_d_values, use_flash_attention=self_attn_use_flash_attention, dropout=dropout
            ) if has_attn else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor]=None, condition=None) -> tuple[torch.Tensor, torch.Tensor]:
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, time_embedding)
            x = attn_block(x)
        return self.downsample(x), x

class UpBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 in_channels: int,
                 out_channels: int,
                 time_embedding_dim: int,
                 activation,
                 scale_factor,
                 self_attn_class,
                 cross_attn_class,
                 has_attn: bool=False,
                 has_condition: bool=False,
                 num_res_blocks: int=2,
                 dropout: float=0.1,
                 self_attn_n_heads=6,
                 self_attn_d_queries=64,
                 self_attn_d_values=64,
                 self_attn_use_flash_attention=True,
                 cross_attn_n_heads=6,
                 cross_attn_d_queries=64,
                 cross_attn_d_values=64,
                 cross_attn_use_flash_attention=True):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels*2 if i == 0 else out_channels, out_channels, time_embedding_dim, activation, dropout)
            for i in range(num_res_blocks)
        ])

        self.attn_blocks = nn.ModuleList([
            self_attn_class(
                out_channels, self_attn_n_heads, self_attn_d_queries, self_attn_d_values, use_flash_attention=self_attn_use_flash_attention, dropout=dropout
            ) if has_attn else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self.cross_attn_blocks = nn.ModuleList([
            cross_attn_class(
                out_channels, cross_attn_n_heads, cross_attn_d_queries, cross_attn_d_values, context_dim=hidden_size, use_flash_attention=cross_attn_use_flash_attention, dropout=dropout
            ) if has_attn and has_condition else nn.Identity()
            for _ in range(num_res_blocks)
        ])

    def forward(self, x: torch.Tensor, skip: list[torch.Tensor], time_embedding: torch.Tensor, condition=None) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for res_block, attn_block, cross_attn_block in zip(self.res_blocks, self.attn_blocks, self.cross_attn_blocks):
            x = res_block(x, time_embedding)
            x = attn_block(x)
            if condition is not None and not isinstance(cross_attn_block, nn.Identity):
                x = cross_attn_block(x, condition)
        return x

class UNet(nn.Module):
    def __init__(
            self,
            hidden_size,
            activation,
            self_attn_class,
            cross_attn_class,
            scale_factor: Union[int, tuple[int, int]] = 2,
            stride: Union[int, tuple[int, int]] = 1,
            in_channels: int = 3,
            model_channels: int = 64,
            out_channels: int = 3,
            channel_multipliers: list[int] = [2, 4, 8],
            time_embedding_dim: int = 256,
            attention_levels: list[bool] = [False, False, True, True],
            num_res_blocks: int = 2,
            dropout: float = 0.1,
            has_condition: bool = False,
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

        activation_type = megatransformer_utils.get_activation_type(activation)
        self.time_embedding = nn.Sequential(
            megatransformer_modules.SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embedding_dim),
            activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(time_embedding_dim),
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
                    has_attn=attention_levels[i],
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    self_attn_n_heads=down_block_self_attn_n_heads,
                    self_attn_d_queries=down_block_self_attn_d_queries,
                    self_attn_d_values=down_block_self_attn_d_values,
                    self_attn_use_flash_attention=down_block_self_attn_use_flash_attention,
                )
            )

        self.middle_res_block = ResidualBlock(
            channels[-1], channels[-1], time_embedding_dim, activation, dropout
        )
        self.middle_attn_block = AudioDiffusionSelfAttentionBlock(
            channels[-1], down_block_self_attn_n_heads, down_block_self_attn_d_queries, down_block_self_attn_d_values, use_flash_attention=down_block_self_attn_use_flash_attention, dropout=dropout
        )
        self.middle_res_block2 = ResidualBlock(
            channels[-1], channels[-1], time_embedding_dim, activation, dropout
        )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channel_multipliers))):
            self.up_blocks.append(
                UpBlock(
                    hidden_size,
                    channels[i + 1],
                    channels[i],
                    time_embedding_dim,
                    activation,
                    scale_factor,
                    self_attn_class,
                    cross_attn_class,
                    has_attn=attention_levels[i],
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    has_condition=has_condition,
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
            model_channels*2, model_channels, time_embedding_dim, activation, dropout
        )
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, condition=None) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)

        h = self.init_conv(x)
        initial_h = h

        skips = []
        for i, down_block in enumerate(self.down_blocks):
            h, skip = down_block(h, time_embedding, condition=condition)
            skips.append(skip)
        
        h = self.middle_res_block(h, time_embedding)
        h = self.middle_attn_block(h)
        h = self.middle_res_block2(h, time_embedding)

        for i, (up_block, skip) in enumerate(zip(self.up_blocks, reversed(skips))):
            h = up_block(h, skip, time_embedding, condition=condition)

        h = torch.cat([h, initial_h], dim=1)
        h = self.final_res_block(h, time_embedding)
        h = self.final_conv(h)

        return h

class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            hidden_size,
            activation,
            scale_factor,
            stride,
            self_attn_class,
            cross_attn_class,
            in_channels: int,
            model_channels: int,
            out_channels: int,
            time_embedding_dim: int,
            num_res_blocks: int,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            num_timesteps: int = 1000,
            predict_epsilon: bool = True,
            has_condition: bool = False,
            unet_dropout: float = 0.1,
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

        self.num_timesteps = num_timesteps
        self.predict_epsilon = predict_epsilon

        self.unet = UNet(
            hidden_size,
            activation,
            scale_factor=scale_factor,
            stride=stride,
            self_attn_class=self_attn_class,
            cross_attn_class=cross_attn_class,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            time_embedding_dim=time_embedding_dim,
            num_res_blocks=num_res_blocks,
            dropout=unet_dropout,
            has_condition=has_condition,
            down_block_self_attn_n_heads=down_block_self_attn_n_heads,
            down_block_self_attn_d_queries=down_block_self_attn_d_queries,
            down_block_self_attn_d_values=down_block_self_attn_d_values,
            down_block_self_attn_use_flash_attention=down_block_self_attn_use_flash_attention,
            up_block_self_attn_n_heads=up_block_self_attn_n_heads,
            up_block_self_attn_d_queries=up_block_self_attn_d_queries,
            up_block_self_attn_d_values=up_block_self_attn_d_values,
            up_block_self_attn_use_flash_attention=up_block_self_attn_use_flash_attention,
            cross_attn_n_heads=cross_attn_n_heads,
            cross_attn_d_queries=cross_attn_d_queries,
            cross_attn_d_values=cross_attn_d_values,
            cross_attn_use_flash_attention=cross_attn_use_flash_attention,
        )

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

        posterior_variance = betas * (1. - alphas_cumprod.clone().detach()) / (1. - alphas_cumprod.clone().detach())
        posterior_variance = torch.cat([torch.tensor([0.0]), posterior_variance])
        self.register_buffer("posterior_variance", posterior_variance)

        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod.clone().detach()) / (1. - alphas_cumprod.clone().detach())
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)

        posterior_mean_coef2 = (1. - alphas_cumprod.clone().detach()) * torch.sqrt(alphas_cumprod.clone().detach()) / (1. - alphas_cumprod.clone().detach())
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        sqrt_recip1m_alphas_cumprod = torch.sqrt(1.0 / (1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip1m_alphas_cumprod", sqrt_recip1m_alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor]=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index, condition=None):
        """Single step of the reverse diffusion process with conditioning"""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        
        # Model forward pass with condition
        model_output = self.unet(x, t, condition=condition)
        
        if self.predict_epsilon:
            # Model predicts noise ε
            pred_epsilon = model_output
            pred_x0 = sqrt_recip_alphas_t * x - sqrt_recip_alphas_t * sqrt_one_minus_alphas_cumprod_t * pred_epsilon
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            posterior_mean = (
                x * (1 - betas_t) / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape)) +
                pred_x0 * betas_t / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape))
            )
        else:
            # Model directly predicts x_0
            pred_x0 = model_output
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            posterior_mean = (
                x * (1 - betas_t) / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape)) +
                pred_x0 * betas_t / torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape))
            )
        
        # Calculate posterior variance using betas_t
        posterior_variance = betas_t * (1 - self._extract(self.alphas_cumprod, t-1, x.shape)) / (1 - self._extract(self.alphas_cumprod, t, x.shape))
        
        if t_index == 0:
            # No noise at the last step (t=0)
            return posterior_mean
        else:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from noise prediction"""
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recip1m_alphas_cumprod_t = self._extract(self.sqrt_recip1m_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recip1m_alphas_cumprod_t * noise

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        b, *_ = t.shape
        out = a.gather(-1, t.long())
        return out.reshape(b, *((1,) * (len(shape) - 1))).to(t.device)
    
    @torch.no_grad()
    def sample(self, device, batch_size: int, image_size: int) -> torch.Tensor:
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

        for time_step in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, time_step)

        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)

        return x
    
    def forward(self, x_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        model_output = self.unet(x_t, t)

        if self.predict_epsilon:
            return model_output, noise
        else:
            return model_output, x_0
