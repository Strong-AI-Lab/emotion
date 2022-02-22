from collections import OrderedDict
from typing import Dict, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtl import MTLModel, MTLTaskConfig


def make_hidden(sizes: Sequence[int], activation: Type[nn.Module]) -> nn.Module:
    if len(sizes) == 0:
        return nn.Identity()
    return nn.Sequential(
        *(
            nn.Sequential(
                nn.Linear(sizes[i], sizes[i + 1]),
                activation(),
            )
            for i in range(len(sizes) - 1)
        )
    )


class SimpleMTLModel(MTLModel):
    def __init__(
        self,
        tasks: Dict[str, MTLTaskConfig],
        input_dim: int,
        opt_lr: float = 0.001,
        shared_units: Sequence[int] = [512],
        task_units: Sequence[int] = [512],
    ):
        super().__init__(tasks=tasks)

        self.opt_lr = opt_lr

        self.save_hyperparameters("opt_lr", "shared_units", "task_units")

        shared_sizes = [input_dim] + list(shared_units)
        self.shared = make_hidden(shared_sizes, nn.ReLU)
        self.add_module("shared", self.shared)

        self.task_layer = {}
        self.task_output = {}
        task_sizes = [shared_sizes[-1]] + list(task_units)
        for name, task in self.tasks.items():
            self.task_layer[name] = make_hidden(task_sizes, nn.ReLU)
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

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        x = F.relu(self.shared(x))
        task_embeddings = {k: self.task_layer[k](x) for k in self.tasks}
        if kwargs.pop("embeddings_only", False):
            return task_embeddings
        out = {k: F.relu(v) for k, v in task_embeddings.items()}
        return {k: self.task_output[k](out[k]) for k in self.tasks}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.opt_lr)
