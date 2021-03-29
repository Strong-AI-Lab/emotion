from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Wrapper
from tensorflow.keras.models import Model

TFModelFunction = Callable[..., Model]
DataFunction = Callable[..., tf.data.Dataset]


def test_fit(
    model_fn: TFModelFunction,
    input_size: Tuple[int],
    *args,
    batch_size: int = 64,
    num_instances: int = 7000,
    **kwargs,
):
    """Tests the given model architecture/structure by training it on
    dummy data.

    Args:
    -----
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

    model = model_fn(*args, n_classes=7, **kwargs)
    model.compile(
        loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"]
    )
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


def create_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Returns a TensorFlow Dataset instance with the given x and y.

    Args:
    -----
    x: numpy.ndarray
        A 2- or 3-D data matrix of shape (n_instances, n_features) or
        (n_instances, seq_len, n_features).
    y: numpy.ndarray
        A 1-D array of length n_instances containing numeric class
        labels.
    sample_weight: numpy.ndarray, optional
        A 1-D array of length n_instances containing sample weights.
        Added as third item in dataset if present.
    batch_size: int
        The batch size to use.
    shuffle: boolean
        Whether or not to shuffle the dataset. Note that shuffling is
        done *before* batching, unlike in `create_tf_dataset_ragged()`.
    """
    if sample_weight is None:
        data = tf.data.Dataset.from_tensor_slices((x, y))
    else:
        data = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))

    if shuffle:
        data = data.shuffle(len(x))
    return data.batch(batch_size).prefetch(8)


def create_tf_dataset_ragged(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Returns a TensorFlow Dataset instance from the ragged x and y.

    Args:
    -----
    x: numpy.ndarray
        A 3-D data matrix of shape (n_instances, length[i], n_features)
        with variable length axis 1.
    y: numpy.ndarray
        A 1-D array of length n_instances containing numeric class
        labels.
    sample_weight: numpy.ndarray, optional
        A 1-D array of length n_instances containing sample weights.
        Added as third item in dataset if present.
    batch_size: int
        The batch size to use.
    shuffle: boolean
        Whether or not to shuffle the dataset. Note that shuffling is
        done *after* batching, because sequences are sorted by length,
        then batched in similar lengths.
    """

    def ragged_to_dense(x: tf.RaggedTensor, y):
        return x.to_tensor(), y

    def ragged_to_dense_weighted(x: tf.RaggedTensor, y, sample_weight):
        return x.to_tensor(), y, sample_weight

    # Sort according to length
    perm = np.argsort([len(a) for a in x])
    x = x[perm]
    y = y[perm]
    if sample_weight is not None:
        sample_weight = sample_weight[perm]

    ragged = tf.RaggedTensor.from_row_lengths(
        np.concatenate(list(x)), [len(a) for a in x]
    )
    if sample_weight is None:
        data = tf.data.Dataset.from_tensor_slices((ragged, y))
    else:
        data = tf.data.Dataset.from_tensor_slices((ragged, y, sample_weight))

    # Group similar lengths in batches, then shuffle batches
    data = data.batch(batch_size)
    if shuffle:
        data = data.shuffle(len(x) // batch_size + 1)

    if sample_weight is not None:
        data = data.map(ragged_to_dense_weighted)
    else:
        data = data.map(ragged_to_dense)
    return data


def print_linear_model_structure(model: Layer, depth: int = 0):
    """Prints the structure of a "sequential" model by listing the layer
    types and shapes in order.

    Args:
    -----
    model: Model
        The model to describe.
    """
    indent = "\t" * depth
    if not isinstance(model, Model):
        print(indent, model.name, model.output_shape)
        return

    for layer in model.layers:
        name = layer.name
        if name.startswith("tf_op_layer_"):
            name = name[12:]

        print(indent, layer.name, layer.output_shape)
        if isinstance(layer, Model):
            print_linear_model_structure(layer, depth + 1)
        elif isinstance(layer, Wrapper):
            print_linear_model_structure(layer.layer, depth + 1)
