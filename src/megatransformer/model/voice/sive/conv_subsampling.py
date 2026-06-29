from typing import Literal
import torch
import torch.nn as nn

from megatransformer.model.norms import RMSNorm


class SpatialLayerNorm2d(nn.Module):
    """LayerNorm across the channel dim at each (F, T) position.

    Pad-invariant by construction: each spatial position normalizes only its
    own C channels, so pad-region values cannot influence valid-region
    statistics. Matches the per-position LayerNorm used in wav2vec2 / HuBERT
    conv stacks.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T] -> [B, F, T, C] -> norm -> [B, C, F, T]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class SpatialLayerNorm1d(nn.Module):
    """LayerNorm across the channel dim at each time position (1d analogue of
    SpatialLayerNorm2d). Pad-invariant: each position normalizes only its own
    C channels."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C] -> norm -> [B, C, T]
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()


class SpatialRMSNorm2d(nn.Module):
    """RMSNorm across the channel dim at each (F, T) position — SpatialLayerNorm2d
    without mean-subtraction (no inter-channel competition from centering).
    Pad-invariant: each spatial position normalizes only its own C channels."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T] -> [B, F, T, C] -> norm (over last dim) -> [B, C, F, T]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class SpatialRMSNorm1d(nn.Module):
    """RMSNorm across the channel dim at each time position (1d analogue of
    SpatialRMSNorm2d)."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C] -> norm (over last dim) -> [B, C, T]
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()


NormType = Literal['batchnorm', 'instancenorm', 'groupnorm', 'layernorm', 'rmsnorm', 'none']


def _build_norm_1d(norm_type: NormType, num_channels: int) -> nn.Module:
    if norm_type == 'batchnorm':
        return nn.BatchNorm1d(num_channels)
    if norm_type == 'instancenorm':
        return nn.InstanceNorm1d(num_channels, affine=True)
    if norm_type == 'groupnorm':
        return nn.GroupNorm(8, num_channels)
    if norm_type == 'layernorm':
        return SpatialLayerNorm1d(num_channels)
    if norm_type == 'rmsnorm':
        return SpatialRMSNorm1d(num_channels)
    if norm_type in ('none', None):
        return nn.Identity()
    raise ValueError(
        f"Unknown downsample norm_type: {norm_type!r}. Expected one of "
        f"'batchnorm', 'instancenorm', 'groupnorm', 'layernorm', 'rmsnorm', 'none'."
    )


def _build_norm_2d(norm_type: NormType, num_channels: int) -> nn.Module:
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(num_channels)
    if norm_type == 'instancenorm':
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == 'groupnorm':
        return nn.GroupNorm(8, num_channels)  # 8 groups is a common choice
    if norm_type == 'layernorm':
        # Per-position LayerNorm across the channel dim. Pad-invariant
        # (wav2vec2 / HuBERT style). Strictly cleaner than InstanceNorm for
        # variable-length voice — pad regions cannot pollute the statistics
        # used at valid-region positions.
        return SpatialLayerNorm2d(num_channels)
    if norm_type == 'rmsnorm':
        return SpatialRMSNorm2d(num_channels)
    if norm_type in ('none', None):
        return nn.Identity()
    raise ValueError(
        f"Unknown downsample norm_type: {norm_type!r}. Expected one of "
        f"'batchnorm', 'instancenorm', 'groupnorm', 'layernorm', 'rmsnorm', 'none'."
    )


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
        norm_type: NormType = 'instancenorm',
    ):
        super().__init__()

        layers = []
        channels = [in_channels] + [out_channels] * len(kernel_sizes)

        for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
            layers.extend([
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=k, stride=s, padding=k // 2),
                _build_norm_1d(norm_type, channels[i + 1]),
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
        norm_type: NormType = 'instancenorm',
    ):
        super().__init__()

        layers = []
        channels = [1] + [out_channels] * len(kernel_sizes)

        for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=k, stride=s, padding=(k[0] // 2, k[1] // 2)),
                _build_norm_2d(norm_type, channels[i + 1]),
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
