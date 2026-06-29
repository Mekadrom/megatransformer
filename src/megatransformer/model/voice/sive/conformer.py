import torch
import torch.nn as nn
import torch.nn.functional as F

from megatransformer.model.norms import build_seq_norm
from megatransformer.model.voice.sive.conv_subsampling import _build_norm_1d


class ConformerConvModule(nn.Module):
    """
    Conformer-style convolution module for speech processing.

    Architecture: norm (pre-norm) -> Pointwise Conv -> GLU -> Depthwise Conv ->
    conv-norm -> Swish -> Pointwise Conv -> Dropout

    The pre-norm (norm_type, lever 3) and the depthwise-conv norm (conv_norm_type,
    lever 4) are independently configurable. Defaults (layernorm / instancenorm)
    match prior behavior.

    The depthwise separable convolution captures local acoustic patterns that
    self-attention alone may miss, making it particularly effective for speech.

    Reference: "Conformer: Convolution-augmented Transformer for Speech Recognition"
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        conv_norm_type: str = "instancenorm",
    ):
        super().__init__()
        inner_dim = d_model * expansion_factor

        self.norm = build_seq_norm(norm_type, d_model)

        # Pointwise expansion with GLU (doubles channels, GLU halves them)
        self.pointwise_conv1 = nn.Conv1d(d_model, inner_dim * 2, kernel_size=1)

        # Depthwise conv (processes each channel independently)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner_dim,  # Depthwise: each channel has its own filter
        )

        # Norm on the depthwise-conv output [B, inner_dim, T]. Attribute name kept
        # as `instance_norm` for checkpoint-key compatibility regardless of type.
        self.instance_norm = _build_norm_1d(conv_norm_type, inner_dim)
        self.activation = nn.SiLU()  # Swish

        # Pointwise projection back to d_model
        self.pointwise_conv2 = nn.Conv1d(inner_dim, d_model, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        # Pre-norm
        x = self.norm(x)

        # [B, T, D] -> [B, D, T] for conv
        x = x.transpose(1, 2)

        # Pointwise expansion + GLU
        x = self.pointwise_conv1(x)  # [B, inner_dim*2, T]
        x = F.glu(x, dim=1)  # [B, inner_dim, T]

        # Depthwise conv
        x = self.depthwise_conv(x)  # [B, inner_dim, T]
        x = self.instance_norm(x)
        x = self.activation(x)

        # Pointwise projection
        x = self.pointwise_conv2(x)  # [B, D, T]
        x = self.dropout(x)

        # [B, D, T] -> [B, T, D]
        return x.transpose(1, 2)
