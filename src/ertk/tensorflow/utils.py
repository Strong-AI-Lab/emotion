"""Utility functions for TensorFlow models.

.. autosummary::
    :toctree:

    compile_wrap
    test_fit
    init_gpu_memory_growth
    print_linear_model_structure
    TFModelFunction
"""

from functools import wraps
from collections.abc import Callable
from typing import Any, Optional, Union

import keras
import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline

__all__ = [
    "compile_wrap",
    "test_fit",
    "init_gpu_memory_growth",
    "print_linear_model_structure",
    "TFModelFunction",
]

TFModelFunction = Callable[..., Union[keras.Model, Pipeline]]


def compile_wrap(
    model_fn: Optional[TFModelFunction] = None,
    opt_cls: type[keras.Optimizer] = keras.optimizers.Adam,
    opt_kwargs: dict[str, Any] = dict(learning_rate=0.0001),
    metrics: list[Union[str, keras.Metric]] = ["sparse_categorical_accuracy"],
    loss: Union[str, keras.Loss] = "sparse_categorical_crossentropy",
    **compile_kwargs,
):
    """Wrapper that takes a model creation function and gives a new
    function which returns a compiled model with the given compile
    parameters.

    Parameters
    ----------
    model_fn: callable, optional
        A method that returns an uncompiled model.
    opt_cls: type
        The Optimizer class to use.
    opt_kwargs: dict
        Keyword arguments to pass to opt_cls.
    metrics: list
        list of metrics to use.
    loss: Loss
        The loss function to use.
    **compile_kwargs: dict
        Other keyword arguments to pass to the model's compile() method.
    """

    def _wrapper(func: Callable[..., keras.Model]):
        @wraps(func)
        def new_model_fn(*args, **kwargs) -> keras.Model:
            model = func(*args, **kwargs)
            model.compile(
                optimizer=opt_cls(**opt_kwargs),
                metrics=metrics,
                loss=loss,
                **compile_kwargs,
            )
            return model

        return new_model_fn

    if model_fn is not None:
        return _wrapper(model_fn)

    return _wrapper


def test_fit(
    model_fn: TFModelFunction,
    input_size: tuple[int, ...],
    *args,
    batch_size: int = 64,
    num_instances: int = 7000,
    **kwargs,
):
    """Tests the given model architecture/structure by training it on
    dummy data.

    Parameters
    ----------
    model_fn: callable
        Function that returns a Keras model. Called as model_fn(*args,
        **kwargs).
    input_size: tuple of int
        Input shape to the model. This is used to generate dummy data of
        the correct shape.
    *args
        Positional arguments to pass to model_fn().
    batch_size: int
        The batch size to use.
    num_instances: int
        The number of instances to generate.
    **kwargs
        Keyword arguments to pass to model_fn().
    """
    for gpu in tf.config.get_visible_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    compiled_fn = compile_wrap(model_fn)
    model = compiled_fn(*args, n_classes=7, **kwargs)
    model.summary()

    valid = num_instances // 10
    rng = np.random.default_rng()
    x = rng.normal(size=(num_instances,) + input_size)
    y = rng.integers(7, size=num_instances)
    train_data = tf.data.Dataset.from_tensor_slices((x[valid:], y[valid:]))
    train_data = train_data.batch(batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices((x[:valid], y[:valid]))
    valid_data = valid_data.batch(batch_size)
    model.fit(train_data, validation_data=valid_data, epochs=2, verbose=1)


def print_linear_model_structure(model: keras.Model) -> None:
    """Prints the structure of a "sequential" model by listing the layer
    types and shapes in order.

    Parameters
    ----------
    model: Model
        The model to describe.
    """

    def print_inner(model: keras.Layer, depth: int = 0):
        indent = "\t" * depth
        if not isinstance(model, keras.Model):
            print(indent, model.name, model.output_shape)
            return

        for layer in model.layers:
            name = layer.name
            if name.startswith("tf_op_layer_"):
                name = name[12:]

            print(indent, name, layer.output_shape)
            if isinstance(layer, keras.Model):
                print_inner(layer, depth + 1)
            elif isinstance(layer, keras.layers.Wrapper):
                print_inner(layer.layer, depth + 1)

    print_inner(model)


def init_gpu_memory_growth() -> None:
    """Sets TensorFlow to allocate memory on GPU as needed instead of
    all at once.
    """
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
