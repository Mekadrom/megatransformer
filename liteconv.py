from torch import nn

import torch
import torch.nn.functional as F

class LiteConv(nn.Module):
    def __init__(self, args, bias=True):
        super(LiteConv, self).__init__()

        self.args = args

        self.glu = nn.GLU(dim=-1)

        self.padding = args.liteconv_kernel_size // 2

        self.depthwise_weight = nn.Parameter(torch.Tensor(args.d_model // 2, args.liteconv_kernel_size))
        self.pointwise_weight = nn.Parameter(torch.Tensor(args.d_model // 2, args.d_model // 2))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(args.d_model // 2))
        else:
            self.bias = None

        self.weight_dropout = nn.Dropout(args.liteconv_weight_dropout)

        nn.init.xavier_uniform_(self.depthwise_weight)
        nn.init.xavier_uniform_(self.pointwise_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, C = input.size()
        H = self.args.n_heads

        depthwise_weight = self.weight_dropout(self.depthwise_weight)
        pointwise_weight = self.weight_dropout(self.pointwise_weight)

        input = self.glu(input)

        input = input.permute(0, 2, 1).contiguous().view(-1, H, T)
        output = F.conv1d(input, depthwise_weight, padding=self.padding, groups=H)
        output = F.conv1d(output, pointwise_weight, groups=H)
        output = output.view(B, C, T)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        output = output.permute(0, 2, 1).contiguous()

        return output
