import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ertk.config import ERTKConfig


@dataclass
class PyTorchModelConfig(ERTKConfig):
    optimiser: str = "adam"
    opt_params: Dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 1e-3
    n_features: int = -1
    n_classes: int = -1
    loss: str = "cross_entropy"
    loss_args: Dict[str, Any] = field(default_factory=dict)


class ERTKPyTorchModel(pl.LightningModule, ABC):
    config: PyTorchModelConfig

    _config_type: ClassVar[Type[PyTorchModelConfig]]
    _friendly_name: ClassVar[str]
    _registry: ClassVar[Dict[str, Type["ERTKPyTorchModel"]]] = {}

    def __init__(
        self,
        config: PyTorchModelConfig,
    ) -> None:
        super().__init__()

        full_config = OmegaConf.merge(type(self).get_default_config(), config)
        # Inplace merge so that subclass __init__() also gets the full config
        config.merge_with(full_config)

        self.save_hyperparameters(PyTorchModelConfig.to_dictconfig(config))
        self.config = config

    def __init_subclass__(
        cls, fname: str = None, config: Type[PyTorchModelConfig] = None
    ) -> None:
        cls._registry = {}
        if fname and config:
            cls._friendly_name = fname
            cls._config_type = config
            for t in cls.mro()[1:-1]:
                t = cast(Type[ERTKPyTorchModel], t)  # For MyPy
                if not hasattr(t, "_registry"):
                    continue
                if fname in t._registry:
                    prev_cls = t._registry[fname]
                    msg = f"Name {fname} already registered with class {prev_cls}."
                    if prev_cls is cls:
                        warnings.warn(msg)
                    else:
                        raise KeyError(msg)
                t._registry[fname] = cls

    @classmethod
    def get_model_class(cls, name: str) -> Type["ERTKPyTorchModel"]:
        try:
            return cls._registry[name]
        except KeyError as e:
            raise ValueError(f"No model named {name}") from e

    @classmethod
    def make_model(cls, name: str, config: PyTorchModelConfig) -> "ERTKPyTorchModel":
        return cls.get_model_class(name)(config)

    @classmethod
    def get_config_type(cls) -> Type[PyTorchModelConfig]:
        return cls._config_type

    @classmethod
    def valid_models(cls) -> List[str]:
        return list(cls._registry)

    @classmethod
    def get_default_config(cls) -> PyTorchModelConfig:
        return OmegaConf.structured(cls._config_type)

    @abstractmethod
    def forward(self, x, **kwargs):  # type: ignore
        raise NotImplementedError()

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Perform any modifications to the input tensor `x`, which is
        extracted from a batch. This will typically have shape
        (batch_size, features) or (batch_size, timesteps, features).
        """
        return x


class SimpleModel(ERTKPyTorchModel):
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
        config: PyTorchModelConfig,
    ) -> None:
        super().__init__(config)
        loss_fn = {
            "cross_entropy": nn.CrossEntropyLoss,
            "nll": nn.NLLLoss,
            "mse": nn.MSELoss,
        }.get(config.loss)
        if loss_fn is not None:
            self.loss = loss_fn(**config.loss_args)
        self.loss_args = config.loss_args
        self.n_features = config.n_features
        self.optimiser = config.optimiser
        self.opt_params = config.opt_params
        self.learning_rate = config.learning_rate

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
        self.log(f"loss/{name}", loss.item())
        return loss

    def get_outputs_for_batch(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x, y, *sw = batch
        sw = sw[0] if len(sw) > 0 else None
        with torch.no_grad():
            x = self.preprocess_input(x)
        yhat = self.forward(x)
        return x, y, yhat, sw

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, *_ = self.get_outputs_for_batch(batch)
        return self.forward(x)

    def configure_optimizers(self):
        optim_fn = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
            "rmsprop": torch.optim.RMSprop,
        }[self.optimiser]
        return optim_fn(self.parameters(), lr=self.learning_rate, **self.opt_params)


class LightningWrapper(SimpleModel):
    """Wraps a torch Module instance to make it compatible with PyTorch
    Lightning. Assumes that the Module only outputs a single tensor.

    Parameters
    ----------
    model: nn.Module
        The model to wrap.
    config: PyTorchModelConfig
        Passed to `SimpleModel`.
    """

    def __init__(self, model: nn.Module, config: PyTorchModelConfig) -> None:
        super().__init__(config)
        self.model = model

    def forward(self, x, *args, **kwargs):
        if hasattr(self.model, "preprocess_input"):
            x = self.model.preprocess_input(x)
        return self.model.forward(x, *args, **kwargs)


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

    def __init__(self, config: PyTorchModelConfig) -> None:
        config.loss = "cross_entropy"
        config.loss_args = config.loss_args or {}
        config.loss_args["reduction"] = "none"
        super().__init__(config)
        self.n_classes = config.n_classes

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y, yhat, sw = self.get_outputs_for_batch(batch)
        loss = self.log_loss("train", y, yhat, sw)
        self.log_acc("train", y, yhat)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y, yhat, sw = self.get_outputs_for_batch(batch)
        self.log_loss("val", y, yhat, sw)
        acc = self.log_acc("val", y, yhat)
        self.log("hp_metric", acc.item())

    def log_acc(self, name: str, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        acc = (yhat.argmax(1) == y).float().mean()
        self.log(f"acc/{name}", acc.item())
        return acc
