from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import omegaconf
import torch
import torch.nn as nn

from ertk.config import ERTKConfig
from ertk.pytorch.utils import get_loss

from ._base import ERTKPyTorchModel, PyTorchModelConfig


@dataclass
class MTLTaskConfig(ERTKConfig):
    output_dim: int = omegaconf.MISSING
    loss: str = omegaconf.MISSING
    metrics: List[str] = field(default_factory=list)
    weight: float = 1


@dataclass
class MTLModelConfig(PyTorchModelConfig):
    tasks: Dict[str, MTLTaskConfig] = omegaconf.MISSING


class MTLModel(ERTKPyTorchModel, ABC):
    tasks: Dict[str, MTLTaskConfig]
    losses: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    config: MTLModelConfig

    def __init__(self, config: MTLModelConfig) -> None:
        super().__init__(config)
        self.tasks = config.tasks
        self.losses = {n: get_loss(t.loss) for n, t in self.tasks.items()}

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, ys = batch
        outputs = self.forward(x)
        total_loss = 0
        accs = {k: 0 for k in self.tasks}
        losses = {k: 0 for k in self.tasks}
        for k, v in outputs.items():
            loss = self.losses[k](v, ys[k])
            losses[k] = loss
            total_loss = total_loss + loss * self.tasks[k].weight
            y_pred = v.argmax(1)
            accs[k] = torch.sum(y_pred == ys[k]).item() / float(len(ys[k]))
        res = {
            "loss/train": total_loss,
            **{f"acc_{k}/train": accs[k] for k in accs},
            **{f"loss_{k}/train": losses[k] for k in losses},
        }
        self.log_dict(res)
        return total_loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, ys = batch
        outputs = self.forward(x)
        total_loss = 0
        accs = {k: 0 for k in self.tasks}
        losses = {k: 0 for k in self.tasks}
        for k, v in outputs.items():
            # print(k, ys[k])
            loss = self.losses[k](v, ys[k])
            losses[k] += loss
            total_loss = total_loss + loss * self.tasks[k].weight
            y_pred = v.argmax(1)
            accs[k] = torch.sum(y_pred == ys[k]).item() / float(len(ys[k]))
        self.log_dict(
            {
                "loss/valid": total_loss,
                **{f"acc_{k}/valid": accs[k] for k in accs},
                **{f"loss_{k}/valid": losses[k] for k in losses},
            }
        )

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.forward(batch[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
