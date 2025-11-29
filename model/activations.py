import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_in):
        super(SwiGLU, self).__init__()

        self.cast = nn.Linear(d_in // 2, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        x = self.cast(x)
        return x

class Snake(nn.Module):
    """
    Snake activation: x + (1/alpha) * sin^2(alpha * x)
    
    The alpha parameter is learnable per channel, allowing the network
    to adapt the periodicity to different frequency ranges.
    """
    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        # Learnable frequency parameter per channel
        self.alpha = nn.Parameter(torch.full((1, channels, 1), alpha_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, T]
        # sin^2(ax) = (1 - cos(2ax)) / 2, but direct computation is fine
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2
