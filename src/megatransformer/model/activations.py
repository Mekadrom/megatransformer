import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU with internal projection back to input dim (legacy)."""
    def __init__(self, d_in):
        super(SwiGLU, self).__init__()

        self.cast = nn.Linear(d_in // 2, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        x = self.cast(x)
        return x


class SwiGLUSimple(nn.Module):
    """SwiGLU gating only - halves the dimension. Use with external projection."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


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


def get_activation_type(activation_function_name):
    if activation_function_name == 'relu':
        return nn.ReLU
    elif activation_function_name == 'gelu':
        return nn.GELU
    elif activation_function_name == 'elu':
        return nn.ELU
    elif activation_function_name == 'selu':
        return nn.SELU
    elif activation_function_name == 'prelu':
        return nn.PReLU
    elif activation_function_name == 'leaky_relu':
        return nn.LeakyReLU
    elif activation_function_name == 'silu':
        return nn.SiLU
    elif activation_function_name == 'tanh':
        return nn.Tanh
    elif activation_function_name == 'sigmoid':
        return nn.Sigmoid
    elif activation_function_name == 'swiglu':
        return SwiGLU
    elif activation_function_name == 'snake':
        # todo: how to distinguish snake1d and snake2d?
        return Snake
    elif activation_function_name == 'none':
        return nn.Identity
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")
