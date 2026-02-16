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
