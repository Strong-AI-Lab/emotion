"""TensorFlow implementations of various models.

.. autosummary::
    :toctree:

    get_tf_model
    get_tf_model_fn
    TFModelConfig


Models
------
.. autosummary::
    :toctree:

    aldeneh2017
    audeep
    depinto2020
    iskhakova2020
    latif2019
    mlp
    rbm
    transformer
    zhang2019
    zhao2019


Layers
------
.. autosummary::
    :toctree:

    layers
"""

import importlib
from functools import partial

import keras

from ertk.tensorflow.utils import TFModelFunction

from ._base import TFModelConfig

__all__ = ["get_tf_model", "get_tf_model_fn", "TFModelConfig"]


def get_tf_model_fn(name: str, **kwargs) -> TFModelFunction:
    """Get a TensorFlow model creation function.

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


def get_tf_model(name: str, **kwargs) -> keras.Model:
    """Get a TensorFlow model by name.

    Parameters
    ----------
    name: str
        The model name. This should be the name of the module that
        implements the model.
    **kwargs:
        Keyword arguments to provide to the model creation function.

    Returns
    -------
    model: Model
        The uncompiled TensorFlow model.
    """
    return get_tf_model_fn(name, **kwargs)()
