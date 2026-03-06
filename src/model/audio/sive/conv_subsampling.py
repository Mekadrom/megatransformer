from typing import Literal
import torch
import torch.nn as nn


class ConvSubsampling(nn.Module):
    """
    Convolutional subsampling frontend (similar to Conformer/wav2vec2).
    Reduces temporal resolution while extracting local features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list = [5, 3, 3],
        strides: list = [2, 2, 1],
        dropout: float = 0.05,
    ):
        super().__init__()

        layers = []
        channels = [in_channels] + [out_channels] * len(kernel_sizes)

        for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
            layers.extend([
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=k, stride=s, padding=k // 2),
                nn.InstanceNorm1d(channels[i + 1], affine=True),
                nn.GELU(),
                nn.Dropout1d(dropout),
            ])

        self.conv = nn.Sequential(*layers)

        # Compute total stride for length calculation
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, T]
        Returns:
            [B, out_channels, T // total_stride]
        """
        return self.conv(x)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        return (input_length + self.total_stride - 1) // self.total_stride


class Conv2dSubsampling(nn.Module):
    """
    Convolutional subsampling frontend (similar to Conformer/wav2vec2) that also downsamples in frequency.
    Reduces temporal resolution while extracting local features.
    """

    def __init__(
        self,
        out_channels: int,
        n_mels: int = 80,
        kernel_sizes: list = [(5, 5), (5, 3), (5, 3)],
        strides: list = [(2, 2), (2, 2), (1, 1)],
        dropout: float = 0.05,
        norm_type: Literal['batchnorm', 'instancenorm', 'groupnorm'] = 'instancenorm',
    ):
        super().__init__()

        layers = []
        channels = [1] + [out_channels] * len(kernel_sizes)

        for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
            norm = None
            if norm_type == 'batchnorm':
                norm = nn.BatchNorm2d(channels[i + 1])
            elif norm_type == 'instancenorm':
                norm = nn.InstanceNorm2d(channels[i + 1], affine=True)
            elif norm_type == 'groupnorm':
                norm = nn.GroupNorm(8, channels[i + 1])  # 8 groups is a common choice
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=k, stride=s, padding=(k[0] // 2, k[1] // 2)),
                norm if norm is not None else nn.Identity(),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ])

        self.conv = nn.Sequential(*layers)

        # Compute total stride for length calculation
        self.total_stride = (1, 1)
        for s in strides:
            self.total_stride = (self.total_stride[0] * s[0], self.total_stride[1] * s[1])

        freq_out = n_mels // self.total_stride[0]  # e.g., 80 // 8 = 10
        self.proj = nn.Linear(out_channels * freq_out, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, F, T]
        Returns:
            [B, out_channels, F // total_stride[0], T // total_stride[1]]
        """
        # if no channels dimension on input mel specs, add a singleton
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, F, T]
        x = self.conv(x)                       # [B, C, F', T']
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)             # [B, T', C, F']
        x = x.reshape(B, T, C * F)            # [B, T', C * F']
        x = self.proj(x)                      # [B, T', d_model]
        x = x.permute(0, 2, 1)                # [B, d_model, T']
        return x

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        return (input_length + self.total_stride[1] - 1) // self.total_stride[1]
