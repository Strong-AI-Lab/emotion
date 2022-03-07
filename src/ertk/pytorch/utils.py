from dataclasses import dataclass
from typing import Callable, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn


@dataclass
class MTLTaskConfig:
    name: str
    weight: float
    output_dim: int
    loss: nn.Module


class LightningWrapper(pl.LightningModule):
    """Wraps a torch Module instance to make it compatible with PyTorch
    Lightning. Assumes that the Module only outputs a single tensor.

    Parameters
    ----------
    model: nn.Module
        The model to wrap.
    loss: callable
        The loss function. Must take the output and ground truth and
        return a loss tensor.
    optim: torch.optim.Optimizer
        The optimiser to use.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optim_fn: Callable[..., torch.optim.Optimizer],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.loss = loss
        self.optim_fn = optim_fn

    def forward(self, x, *args, **kwargs):
        if hasattr(self.model, "preprocess_input"):
            x = self.model.preprocess_input(x)
        return self.model.forward(x, *args, **kwargs)

    def log_loss(
        self, name: str, y: torch.Tensor, yhat: torch.Tensor, sw: Optional[torch.Tensor]
    ):
        loss = self.loss(yhat, y)
        if sw is not None:
            loss = loss.dot(sw) / sw.sum()
        else:
            loss = loss.mean()
        self.log(f"{name}_loss", loss.item())
        return loss

    def training_step(self, batch, batch_idx: int):
        x, y, *sw = batch
        sw = sw[0] if len(sw) > 0 else None
        yhat = self.forward(x)
        loss = self.log_loss("train", y, yhat, sw)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y, *sw = batch
        sw = sw[0] if len(sw) > 0 else None
        yhat = self.forward(x)
        self.log_loss("val", y, yhat, sw)

    def predict_step(self, batch, batch_idx: int):
        x, *_ = batch
        return self.forward(x)

    def configure_optimizers(self):
        return self.optim_fn(self.parameters())
