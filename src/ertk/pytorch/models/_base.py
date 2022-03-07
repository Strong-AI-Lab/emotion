from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn


class SimpleModel(pl.LightningModule, ABC):
    """A model that takes input x, gives output y, and has a single loss
    function. Each batch is either (x, y) pairs or (x, y, sw) triplets,
    where sw is the sample weights.

    Parameters
    ----------
    n_features: int
        Input dimensionality.
    loss: callable
        A callable that takes predictions yhat and ground truth y to
        yield a loss::
            l = loss(yhat, y)
        It is assumed that if sw is given then the returned loss is a 1D
        tensor of losses per instance in the batch.
    optim_fn: callable
        Function that returns an optimiser for use with this model. The
        default optimiser is `torch.optim.Adam` with default args.
    """

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
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y, yhat, sw = self.get_outputs_for_batch(batch)
        loss = self.log_loss("train", y, yhat, sw)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y, yhat, sw = self.get_outputs_for_batch(batch)
        self.log_loss("val", y, yhat, sw)

    def log_loss(
        self, name: str, y: torch.Tensor, yhat: torch.Tensor, sw: Optional[torch.Tensor]
    ) -> torch.Tensor:
        loss = self.loss(yhat, y)
        if sw is not None:
            loss = loss.dot(sw) / sw.sum()
        else:
            loss = loss.mean()
        self.log(f"{name}_loss", loss.item())
        return loss

    def get_outputs_for_batch(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x, y, *sw = batch
        sw = sw[0] if len(sw) > 0 else None
        x = self.preprocess_input(x)
        yhat = self.forward(x)
        return x, y, yhat, sw

    def predict_step(self, batch, batch_idx: int):
        x, *_ = batch
        return self.forward(x)

    @abstractmethod
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        return self.optim_fn(self.parameters())


class SimpleClassificationModel(SimpleModel):
    """A classification model that takes input x, gives output
    probabilities y, and has a single loss function. Each batch is
    either (x, y) pairs or (x, y, sw) triplets, where sw is the sample
    weights.

    Parameters
    ----------
    n_classes: int
        Number of output classes.
    kwargs: dict
        Other keyword arguments to pass to `SimpleModel.__init__()`.
    """

    def __init__(self, n_classes: int, **kwargs) -> None:
        super().__init__(loss=nn.CrossEntropyLoss(reduction="none"), **kwargs)
        self.n_classes = n_classes

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y, yhat, sw = self.get_outputs_for_batch(batch)
        loss = self.log_loss("train", y, yhat, sw)
        self.log_acc("train", y, yhat)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y, yhat, sw = self.get_outputs_for_batch(batch)
        self.log_loss("val", y, yhat, sw)
        self.log_acc("val", y, yhat)

    def log_acc(self, name: str, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        acc = torch.sum(yhat.argmax(1) == y) / float(len(y))
        self.log(f"{name}_acc", acc.item())
        return acc
