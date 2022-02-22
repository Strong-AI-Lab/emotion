from abc import ABC, abstractmethod
from typing import Callable, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(pl.LightningModule, ABC):
    def __init__(
        self,
        n_features: int,
        loss: Callable[..., torch.Tensor],
        optim_fn: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.loss = loss
        self.n_features = n_features
        self.optim_fn = optim_fn

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y, *sw = batch
        with torch.no_grad():
            x = self.preprocess_input(x)
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        if len(sw) > 0 and sw[0] is not None:
            loss = loss.dot(sw[0])
        else:
            loss = loss.mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y, *sw = batch
        with torch.no_grad():
            x = self.preprocess_input(x)
        yhat = self.forward(x)
        loss = self.loss(yhat, y)
        if len(sw) > 0 and sw[0] is not None:
            loss = loss.dot(sw[0]).item()
        else:
            loss = loss.mean().item()
        acc = torch.sum(yhat.argmax(1) == y).item() / float(len(y))
        self.log_dict({"val_loss": loss, "val_acc": acc})

    def predict_step(self, batch, batch_idx: int):
        x, *_ = batch
        return self.forward(x)

    @abstractmethod
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        return self.optim_fn(self.parameters())


class SimpleClassificationModel(SimpleModel):
    def __init__(self, n_classes: int, **kwargs) -> None:
        super().__init__(loss=nn.CrossEntropyLoss(reduction="none"), **kwargs)
        self.n_classes = n_classes
