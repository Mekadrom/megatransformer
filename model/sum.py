from torch import nn

import torch

class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y
