import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union


class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Mult(nn.Module):
    def __init__(self):
        super(Mult, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embeddings for diffusion models."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, dtype=time.dtype)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=time.dtype, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))

        return embeddings


class AvgMaxAdaptivePool2d(nn.Module):
    def __init__(self, output_size: Union[int, tuple[int, int]]=(1, 1)):
        super(AvgMaxAdaptivePool2d, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        elif isinstance(output_size, tuple) and len(output_size) == 1:
            output_size = (output_size[0], output_size[0])
        elif not isinstance(output_size, tuple) or len(output_size) != 2:
            raise ValueError("output_size must be an int or a tuple of two ints.")
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, self.output_size)
        max_pool = F.adaptive_max_pool2d(x, self.output_size)
        # Concatenate along the channel dimension
        return torch.cat((avg_pool, max_pool), dim=1)
