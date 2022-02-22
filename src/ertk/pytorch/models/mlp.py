from typing import Sequence

from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        units: Sequence[int] = [512],
        dropout: float = 0.5,
    ):
        super().__init__()

        sizes = [n_features] + list(units)
        self.hidden = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(), nn.Dropout(dropout)
                )
                for i in range(len(units))
            )
        )
        self.final = nn.Linear(units[-1], n_classes)

    def forward(self, x):
        return self.final(self.hidden(x))
