from abc import ABC, abstractmethod
from typing import Dict

import pytorch_lightning as pl
import torch

from ertk.pytorch.utils import MTLTaskConfig


class MTLModel(pl.LightningModule, ABC):
    tasks: Dict[str, MTLTaskConfig]

    def __init__(self, tasks: Dict[str, MTLTaskConfig], **kwargs) -> None:
        super().__init__(**kwargs)

        self.tasks = tasks

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, ys = batch
        outputs = self.forward(x)
        loss = 0
        losses = {k: 0 for k in self.tasks}
        for k, v in outputs.items():
            losses[k] += self.tasks[k].loss(v, ys[k]) * self.tasks[k].weight
            loss += losses[k]
        res = dict(loss=loss, **{f"train_{k}_loss": losses[k].detach() for k in losses})
        self.log_dict(res)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, ys = batch
        outputs = self.forward(x)
        val_loss = 0
        accs = {k: 0 for k in self.tasks}
        losses = {k: 0 for k in self.tasks}
        for k, v in outputs.items():
            losses[k] += self.tasks[k].loss(v, ys[k]).item() * self.tasks[k].weight
            val_loss += losses[k]
            y_pred = v.argmax(1)
            accs[k] = torch.sum(y_pred == ys[k]).item() / float(len(ys[k]))
        self.log_dict(
            dict(
                val_loss=val_loss,
                **{f"{k}_acc": accs[k] for k in accs},
                **{f"{k}_loss": losses[k] for k in losses},
            )
        )
