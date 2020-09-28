from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def create_tf_dataset(x: np.ndarray, y: np.ndarray,
                      sample_weight: Optional[np.ndarray] = None,
                      batch_size: int = 64,
                      shuffle: bool = True) -> tf.data.Dataset:
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


def create_tf_dataset_ragged(x: np.ndarray, y: np.ndarray,
                             sample_weight: Optional[np.ndarray] = None,
                             batch_size: int = 64,
                             shuffle: bool = True) -> tf.data.Dataset:
    """Returns a TensorFlow Dataset instance from the ragged x and y.

    x , while y is a 1-D array of
    length n_instances.

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
    def ragged_to_dense(x, y):
        return x.to_tensor(), y

    # Sort according to length
    perm = np.argsort([len(a) for a in x])
    x = x[perm]
    y = y[perm]
    if sample_weight is not None:
        sample_weight = sample_weight[perm]

    ragged = tf.RaggedTensor.from_row_lengths(np.concatenate(list(x)),
                                              [len(a) for a in x])
    if sample_weight is None:
        data = tf.data.Dataset.from_tensor_slices((ragged, y))
    else:
        data = tf.data.Dataset.from_tensor_slices((ragged, y, sample_weight))
    # Group similar lengths in batches, then shuffle batches
    data = data.batch(batch_size)
    if shuffle:
        data = data.shuffle(len(x) // batch_size + 1)
    return data.map(ragged_to_dense)


def print_linear_model_structure(model: Model):
    """Prints the structure of a "sequential" model by listing the layer
    types and shapes in order.

    Args:
    -----
    model: Model
        The model to describe.
    """
    for layer in model.layers:
        name = layer.name
        if name.startswith('tf_op_layer_'):
            name = name[12:]
        print(layer.name, layer.output_shape)
