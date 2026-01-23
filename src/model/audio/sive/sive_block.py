import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional

from rotary_embedding_torch import RotaryEmbedding

from model import activations
from model.activations import get_activation_type
from model.audio.sive.conformer import ConformerConvModule
from model.head_dropout import HeadDropout


class SpeakerInvariantVoiceEncoderBlock(nn.Module):
    """
    Transformer encoder block with pre-norm, optional RoPE, Conformer conv, and configurable activation.

    Supports:
    - Pre-norm architecture (more stable training)
    - Rotary Position Embeddings (RoPE) for better position modeling
    - Conformer-style convolution module for local acoustic patterns
    - Macaron-style FFN (half-step FFN before and after attention)
    - SwiGLU or GELU activation in FFN
    - DropHead regularization
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        head_drop_prob: float = 0.0,
        # Architectural options
        conformer_kernel_size: int = 31,
        activation: str = "gelu",
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        # Rotary position embeddings
        self.rotary_embedding = RotaryEmbedding(dim=self.head_dim)
        # custom attention with RoPE (can't use nn.MultiheadAttention)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Head dropout: drops entire attention heads for regularization
        self.head_dropout = HeadDropout(
            num_heads=n_heads,
            head_dim=self.head_dim,
            drop_prob=head_drop_prob,
        ) if head_drop_prob > 0 else nn.Identity()

        # Conformer conv module (between attention and FFN)
        self.conv_module = ConformerConvModule(
            d_model=d_model,
            kernel_size=conformer_kernel_size,
            dropout=dropout,
        )

        # Helper to build FFN
        def build_ffn():
            activation_type = get_activation_type(activation)
            if activation_type == activations.SwiGLU:
                return nn.Sequential(
                    nn.Linear(d_model, d_ff * 2),
                    activations.SwiGLUSimple(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                )
            else:
                return nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    activation_type(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                )

        # Macaron-style: two half-step FFNs (each scaled by 0.5)
        self.ff1 = build_ffn()  # First half-step FFN
        self.ff2 = build_ffn()  # Second half-step FFN
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.norm_ff2 = nn.LayerNorm(d_model)

        self.norm1 = nn.LayerNorm(d_model)  # For attention
        self.norm_conv = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _rope_attention(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention with rotary position embeddings."""
        B, T, D = x.shape

        # Project to Q, K, V
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        # obtain head dimension
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D_h]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        q = self.rotary_embedding.rotate_queries_or_keys(q)
        k = self.rotary_embedding.rotate_queries_or_keys(k)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, T, T]

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, T] True for padded positions
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, T]
                float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, T, D_h]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]

        return self.o_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            key_padding_mask: [B, T] True for padded positions
        Returns:
            [B, T, D]

        Macaron-style: ½FFN → Attn → Conv → ½FFN
        """
        # macaron: first half-step FFN (scaled by 0.5)
        residual = x
        x = self.norm_ff1(x)
        x = 0.5 * self.ff1(x)
        x = residual + x

        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x = self._rope_attention(x, key_padding_mask=key_padding_mask)
        x = self.head_dropout(x)  # DropHead: randomly drop entire attention heads
        x = self.dropout(x)
        x = residual + x

        # conformer conv
        residual = x
        x = self.conv_module(x)
        x = residual + x

        # macaron: second half-step FFN (scaled by 0.5)
        residual = x
        x = self.norm_ff2(x)
        x = 0.5 * self.ff2(x)
        x = residual + x

        return x
