from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn

from ._base import PyTorchModelConfig, SimpleClassificationModel


@dataclass
class MLPConfig(PyTorchModelConfig):
    units: List[int] = field(default_factory=lambda: [512])
    dropout: float = 0.5


class Model(SimpleClassificationModel):
    def __init__(self, config: MLPConfig):
        super().__init__(config)

        sizes = [config.n_features] + list(config.units)
        self.hidden = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                )
                for i in range(len(config.units))
            )
        )
        self.final = nn.Linear(config.units[-1], config.n_classes)

    def forward(self, x: torch.Tensor):  # type: ignore
        return self.final(self.hidden(x))
