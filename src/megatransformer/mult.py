from torch import nn

import torch

class Mult(nn.Module):
    def __init__(self):
        super(Mult, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y
