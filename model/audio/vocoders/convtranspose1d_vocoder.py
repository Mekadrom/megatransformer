import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import Literal

from model.activations import Snake


class ConvTranspose1DVocoderUpsampleBlock(nn.Module):
    """Upsampling block for the vocoder."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int,
        kernel_size: int = 8,
        activation_fn: Literal["leaky_relu", "snake"] = 'leaky_relu',
        snake_alpha_init: float = 1.0,
        leaky_relu_slope: float = 0.1,
        norm_type: Literal['batch', 'weight', 'group'] = 'weight'
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.leaky_relu_slope = leaky_relu_slope
        
        # Transposed convolution for upsampling
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=upsample_factor,
            padding=(kernel_size - upsample_factor) // 2
        )

        match activation_fn:
            case 'leaky_relu':
                self.act1 = nn.LeakyReLU(negative_slope=leaky_relu_slope)
            case 'snake':
                self.act1 = Snake(out_channels, snake_alpha_init)
            case _:
                raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.norm: nn.Module
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
            self._init_weights()
        elif norm_type == 'weight':
            self.conv = weight_norm(self.conv)
        elif norm_type == 'group':
            # num_groups=32 is common, but adjust if out_channels isn't divisible
            num_groups = min(32, out_channels)
            self.norm = nn.GroupNorm(num_groups, out_channels)
            self._init_weights()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def _init_weights(self):
        if self.activation_fn == 'snake':
            nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        else:
            nn.init.kaiming_normal_(self.conv.weight, a=self.leaky_relu_slope, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.conv(features)
        if hasattr(self, "norm"):
            features = self.norm(features)
        features = self.act1(features)
        return features
