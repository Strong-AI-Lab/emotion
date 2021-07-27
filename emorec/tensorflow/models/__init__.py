"""TensorFlow implementations of various models."""

import importlib
from typing import Callable, Optional

from tensorflow.keras.models import Model

from .audeep import audeep_trae
from .rbm import BBRBM, DBN, DecayType


def get_tf_model_fn(name: str) -> Callable[..., Model]:
    """Get a TensorFlow model creation function.

    Args:
    -----
    name: str
        The model name. This should be the name of the module that
        implements the model.

    Returns:
    --------
    model_fn: callable
        A method that takes arguments and returns a Model instance.
    """
    module = importlib.import_module(f".{name}", __package__)
    model_fn = getattr(module, "model")
    return model_fn


def get_tf_model(
    name: str, n_features: int, n_classes: Optional[int] = None, **kwargs
) -> Model:
    """Get a TensorFlow model by name.

    Args:
    -----
    name: str
        The model name. This should be the name of the module that
        implements the model.
    n_features: int
        The number of features used for input. For a raw waveform,
        n_features = 1.
    n_classes: int, optional
        The number of classes to output for classification models.
    **kwargs:
        Other keyword arguments to provide to the model creation
        function.

    Returns:
    --------
    model: Model
        The uncompiled TensorFlow model.
    """
    model_fn = get_tf_model_fn(name)
    if n_features > 1:
        if n_classes is not None:
            model = model_fn(n_features, n_classes, **kwargs)
        else:
            model = model_fn(n_features, **kwargs)
    else:
        if n_classes is not None:
            model = model_fn(n_classes, **kwargs)
        else:
            model = model_fn(**kwargs)
    return model
