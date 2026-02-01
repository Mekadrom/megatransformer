import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block - efficient and effective for audio.
    Depthwise conv -> pointwise expand -> pointwise contract
    """
    def __init__(self, dim, ovr_out_dim=None, kernel_size=7, expansion=4):
        super().__init__()
        self.ovr_out_dim = ovr_out_dim

         # depthwise conv1d
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
        
        # pointwise (linear is good enough)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, ovr_out_dim if ovr_out_dim is not None else dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, T, C] for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # [B, C, T]
        
        if self.ovr_out_dim is not None:
            return x  # no residual if changing channels
        return residual + x
