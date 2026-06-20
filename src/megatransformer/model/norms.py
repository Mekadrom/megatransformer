import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sqr = x**2
        RMS = torch.rsqrt(x_sqr.mean(dim = -1, keepdim = True) + self.eps)
        new_x = x * RMS
        new_x = new_x * self.weight

        return new_x


def create_norm(hidden_size, norm_type, norm_eps):
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=norm_eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(hidden_size, eps=norm_eps)
    else:
        raise Exception(f"Unknown normalization type {norm_type}")
