import math

import torch
import torch.nn.functional as F
from torch import nn

nn.init.normal_


class Attention1D(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_channels, 1) / math.sqrt(in_channels))

    def forward(self, x: torch.Tensor):
        # x : (..., length, in_channels)
        alpha = F.softmax(x.matmul(self.weight), -2)
        # alpha : (..., length, 1)
        r = torch.sum(alpha * x, -2)
        return r
