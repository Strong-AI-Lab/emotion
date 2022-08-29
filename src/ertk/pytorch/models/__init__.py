import importlib
from functools import partial
from typing import Callable

from torch.nn import Module

from . import aldeneh2017, mlp, wav2vec_ft, zhang2019
from ._base import (
    ERTKPyTorchModel,
    LightningWrapper,
    PyTorchModelConfig,
    SimpleClassificationModel,
    SimpleModel,
)
from .layers import Attention1D, make_fc


def get_pt_model_fn(name: str, **kwargs) -> Callable[..., Module]:
    """Get a PyTorch model creation function.

    Parameters
    ----------
    name: str
        The model name. This should be the name of the module that
        implements the model.
    **kwargs:
        Keyword arguments to provide to the model creation function.

    Returns
    -------
    model_fn: callable
        A method that takes arguments and returns a Module instance.
    """
    try:
        module = importlib.import_module(f".{name}", __package__)
        model_fn = getattr(module, "Model")
    except (ImportError, AttributeError) as e:
        raise ValueError(f"{name} does not correspond to a valid PyTorch model.") from e
    return partial(model_fn, **kwargs)


def get_pt_model(name: str, **kwargs) -> Module:
    """Get a PyTorch model by name.

    Parameters
    ----------
    name: str
        The model name. This should be the name of the module that
        implements the model.
    **kwargs:
        Keyword arguments to provide to the model creation function.

    Returns
    -------
    model: Module
        The PyTorch model.
    """
    return get_pt_model_fn(name, **kwargs)()
