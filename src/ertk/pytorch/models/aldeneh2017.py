from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from ._base import PyTorchModelConfig, SimpleClassificationModel


@dataclass
class Aldeneh2017Config(PyTorchModelConfig):
    conv_dims: List[int] = field(default_factory=lambda: [384] * 4)
    kernel_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    dense_dims: List[int] = field(default_factory=lambda: [1024, 1024])


class Model(SimpleClassificationModel):
    def __init__(self, config: Aldeneh2017Config):
        super().__init__(config)

        if len(config.conv_dims) != len(config.kernel_sizes):
            raise ValueError("`conv_dims` must be the same size as `kernel_sizes`.")

        self.convs = nn.ModuleList()
        for dim, ksize in zip(config.conv_dims, config.kernel_sizes):
            self.convs.append(nn.Conv1d(config.n_features, dim, kernel_size=ksize))

        dense_in = sum(config.conv_dims)
        dense_dims = [dense_in] + list(config.dense_dims)
        self.dense = nn.Sequential()
        for d1, d2 in zip(dense_dims[:-1], dense_dims[1:]):
            self.dense.append(nn.Sequential(nn.Linear(d1, d2), nn.ReLU(inplace=True)))
        self.dense.append(nn.Linear(dense_dims[-1], config.n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        cs = []
        for conv in self.convs:
            c, _ = torch.max(F.relu(conv(x)), 2)
            cs.append(c)
        x = torch.cat(cs, 1).squeeze(1)
        out = self.dense(x)
        return out
