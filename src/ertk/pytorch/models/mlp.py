"""Simple multi-layer perceptron (MLP) model."""

from dataclasses import dataclass, field

import torch

from ._base import PyTorchModelConfig, SimpleClassificationModel
from .layers import make_fc

__all__ = ["MLPModel", "MLPConfig"]


@dataclass
class MLPConfig(PyTorchModelConfig):
    units: list[int] = field(default_factory=lambda: [512])
    dropout: float = 0.5
    activation: str = "relu"


class MLPModel(SimpleClassificationModel, fname="mlp", config=MLPConfig):
    config: MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        sizes = [config.n_features] + list(config.units) + [config.n_classes]
        self.ffn = make_fc(
            sizes, activation=config.activation, dropout=config.dropout, norm="none"
        )

    def forward(self, x: torch.Tensor):  # type: ignore
        return self.ffn(x)
