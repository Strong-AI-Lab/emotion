"""Simple MTL model consisting of shared layers followed by
task-specific layers.
"""

from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_fc
from .mtl import MTLModel, MTLModelConfig

__all__ = ["SimpleMTLModel", "SimpleMTLModelConfig"]


@dataclass
class SimpleMTLModelConfig(MTLModelConfig):
    shared_units: list[int] = field(default_factory=lambda: [512])
    task_units: list[int] = field(default_factory=lambda: [512])


class SimpleMTLModel(MTLModel, fname="simple_mtl", config=SimpleMTLModelConfig):
    config: SimpleMTLModelConfig

    def __init__(self, config: SimpleMTLModelConfig):
        super().__init__(config)

        shared_sizes = [config.n_features] + list(config.shared_units)
        self.shared = make_fc(shared_sizes, activation="relu", dropout=0, norm="none")
        self.shared.append(nn.Tanh())
        self.add_module("shared", self.shared)

        self.task_layer = {}
        self.task_output = {}
        task_sizes = [shared_sizes[-1]] + list(config.task_units)
        for name, task in self.tasks.items():
            self.task_layer[name] = make_fc(task_sizes, activation="relu", dropout=0)
            self.task_layer[name].append(nn.ReLU())
            self.task_output[name] = nn.Linear(task_sizes[-1], task.output_dim)
            submod = nn.Sequential(
                OrderedDict(
                    {
                        "embeddings": self.task_layer[name],
                        "output": self.task_output[name],
                    }
                )
            )
            self.add_module(name, submod)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        x = F.relu(self.shared(x))
        task_embeddings = {k: self.task_layer[k](x) for k in self.tasks}
        if kwargs.pop("embeddings_only", False):
            return task_embeddings
        out = {k: F.relu(v) for k, v in task_embeddings.items()}
        return {k: self.task_output[k](out[k]) for k in self.tasks}
