"""TensorFlow implementations of various models."""

import importlib
from functools import partial

from tensorflow.keras.models import Model

from ertk.tensorflow.utils import TFModelFunction


def get_tf_model_fn(name: str, **kwargs) -> TFModelFunction:
    """Get a TensorFlow model creation function.

    Args:
    -----
    name: str
        The model name. This should be the name of the module that
        implements the model.
    **kwargs:
        Keyword arguments to provide to the model creation function.

    Returns:
    --------
    model_fn: callable
        A method that takes arguments and returns a Model instance.
    """
    try:
        module = importlib.import_module(f".{name}", __package__)
        model_fn = getattr(module, "model")
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"{name} does not correspond to a valid TensorFlow model."
        ) from e
    return partial(model_fn, **kwargs)


def get_tf_model(name: str, **kwargs) -> Model:
    """Get a TensorFlow model by name.

    Args:
    -----
    name: str
        The model name. This should be the name of the module that
        implements the model.
    **kwargs:
        Keyword arguments to provide to the model creation function.

    Returns:
    --------
    model: Model
        The uncompiled TensorFlow model.
    """
    return get_tf_model_fn(name, **kwargs)()
