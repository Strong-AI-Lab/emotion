import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar, cast

import tensorflow as tf
from keras import Model
from omegaconf import OmegaConf

from ertk.config import ERTKConfig

__all__ = ["TFModelConfig", "ERTKTensorFlowModel"]

T = TypeVar("T", bound="TFModelConfig")


@dataclass
class TFModelConfig(ERTKConfig):
    """Configuration for a TensorFlow model."""

    optimiser: str = "adam"
    """The optimiser to use."""
    opt_params: dict[str, Any] = field(default_factory=dict)
    """The parameters to pass to the optimiser."""
    learning_rate: float = 1e-3
    """The learning rate to use."""
    n_features: int = -1
    """The number of features in the input data."""
    n_classes: int = -1
    """The number of classes in the output data."""
    loss: str = "sparse_categorical_crossentropy"
    """The loss function to use."""


class ERTKTensorFlowModel(Model, ABC):
    """Base class for PyTorch models.

    This class is a subclass of
    :class:`pytorch_lightning.LightningModule` and implements the
    :meth:`training_step` method. It also provides a
    :meth:`get_model_class` method for retrieving a model class by name.

    Parameters
    ----------
    config : TFModelConfig
        The model configuration.
    """

    config: TFModelConfig
    """The model configuration."""

    _config_type: ClassVar[type[TFModelConfig]]
    _friendly_name: ClassVar[str]
    _registry: ClassVar[dict[str, type["ERTKTensorFlowModel"]]] = {}

    def __init__(
        self,
        config: TFModelConfig,
    ) -> None:
        super().__init__()

        if OmegaConf.is_config(config):
            full_config = OmegaConf.merge(type(self).get_default_config(), config)
            # Inplace merge so that subclass __init__() also gets the full config
            config.merge_with(full_config)  # type: ignore

        self.config = config

    def __init_subclass__(
        cls, fname: str | None = None, config: type[TFModelConfig] | None = None
    ) -> None:
        cls._registry = {}
        if fname and config:
            cls._friendly_name = fname
            cls._config_type = config
            for t in cls.mro()[1:-1]:
                t = cast(type[ERTKTensorFlowModel], t)  # For MyPy
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

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_compiled_model(cls, config: TFModelConfig) -> "ERTKTensorFlowModel":
        """Get a compiled model.

        Parameters
        ----------
        config : TFModelConfig
            The model configuration.

        Returns
        -------
        ERTKTensorFlowModel
            The compiled model.
        """
        raise NotImplementedError

    @classmethod
    def get_model_class(cls, name: str) -> type["ERTKTensorFlowModel"]:
        """Get a model class by name.

        Parameters
        ----------
        name : str
            The name of the model class to retrieve.

        Returns
        -------
        Type[ERTKTensorFlowModel]
            The model class.

        Raises
        ------
        ValueError
            If no model class with the given name is registered.
        """
        try:
            return cls._registry[name]
        except KeyError as e:
            raise ValueError(f"No model named {name}") from e

    @classmethod
    def make_model(cls, name: str, config: TFModelConfig) -> "ERTKTensorFlowModel":
        """Make a model by name.

        Parameters
        ----------
        name : str
            The name of the model class to retrieve.
        config : TFModelConfig
            The model configuration.

        Returns
        -------
        ERTKTensorFlowModel
            The model created with the given config.
        """
        return cls.get_model_class(name)(config)

    @classmethod
    def get_config_type(cls) -> type[TFModelConfig]:
        """Get the configuration dataclass for this model.

        Returns
        -------
        Type[TFModelConfig]
            The configuration dataclass.

        Notes
        -----
        This is a class method rather than a property because it is
        needed before an instance of the model is created.
        """
        return cls._config_type

    @classmethod
    def valid_models(cls) -> list[str]:
        """Get a list of valid model names.

        Returns
        -------
        List[str]
            A list of valid model names.
        """
        return list(cls._registry)

    @classmethod
    def get_default_config(cls) -> TFModelConfig:
        """Get the default configuration for this model.

        Returns
        -------
        TFModelConfig
            The default configuration.
        """
        return OmegaConf.structured(cls._config_type)

    def preprocess_input(self, x: tf.Tensor) -> tf.Tensor:
        """Perform any modifications to the input tensor `x`, which is
        extracted from a batch. This will typically have shape
        (batch_size, features) or (batch_size, timesteps, features).

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The preprocessed input tensor.

        Notes
        -----
        The default implementation returns `x` unchanged.
        """
        return x
