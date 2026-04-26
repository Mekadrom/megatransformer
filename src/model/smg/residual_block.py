import torch
import torch.nn as nn

from model import activations
from model.activations import get_activation_type


class ResidualBlock1d(nn.Module):
    """
    Residual block for 1D sequential data.

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
            self.act1 = activations.Snake(out_channels)
            self.act2 = activations.Snake(out_channels)
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
            self.act1 = activations.Snake2d(out_channels)
            self.act2 = activations.Snake2d(out_channels)
        else:
            activation_type = get_activation_type(activation_fn)
            if activation_type in [activations.SwiGLU, activations.Snake2d]:
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
