from torch import nn

class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x, y):
        return x + y
