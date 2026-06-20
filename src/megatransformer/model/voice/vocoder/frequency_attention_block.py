import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAttentionBlock(nn.Module):
    """
    Self-attention block for frequency-domain vocoders.

    Applies multi-head attention across the time dimension to help learn
    global phase coherence. Harmonics are related across time, and attention
    allows the model to learn these long-range dependencies.

    Uses pre-norm architecture with residual connection.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.norm = nn.LayerNorm(dim)

        # Single projection for Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input tensor

        Returns:
            [B, C, T] output tensor with attention applied
        """
        B, C, T = x.shape
        residual = x

        # Transpose for LayerNorm: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        x = self.norm(x)

        qkv: torch.Tensor = self.qkv(x)

        # Compute Q, K, V
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        # Transpose back: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        return residual + x
