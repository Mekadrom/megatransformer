import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            offset: Starting position index (for KV-cached generation).
        Returns:
            [B, T, D] with positional encoding added
        """
        x = x + self.pe[:, offset:offset + x.size(1), :]
        return self.dropout(x)


class Sinusoidal2DPositionalEmbedding(nn.Module):
    """2D sinusoidal positional encoding for image patches."""
    def __init__(self, grid_size: int, d_model: int):
        """Build a frozen 2D sinusoidal positional encoding.

        Splits d_model in half: first half encodes row, second half encodes
        column. Returns shape (1, grid_size*grid_size, d_model).
        """

        super().__init__()
        half_d = d_model // 2
        positions = torch.arange(grid_size, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, half_d, 2, dtype=torch.float) * (-math.log(10000.0) / half_d))

        # Row encoding (first half of d_model)
        row_pe = torch.zeros(grid_size, half_d)
        row_pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        row_pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)

        # Column encoding (second half of d_model)
        col_pe = torch.zeros(grid_size, half_d)
        col_pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        col_pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)

        # Combine: for each (row, col) pair, concat row_pe[row] and col_pe[col]
        pe = torch.zeros(grid_size * grid_size, d_model)
        for r in range(grid_size):
            for c in range(grid_size):
                pe[r * grid_size + c, :half_d] = row_pe[r]
                pe[r * grid_size + c, half_d:] = col_pe[c]

        pe = pe.unsqueeze(0)  # (1, n_patches, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_patches, d_model]
        Returns:
            [B, n_patches, d_model] with positional encoding added
        """
        return x + self.pe