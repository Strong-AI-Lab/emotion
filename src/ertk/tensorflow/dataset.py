"""Dataset utilities for TensorFlow.

.. autosummary::
    :toctree:

    tf_dataset_gen
    tf_dataset_mem_ragged
    tf_dataset_mem
    BatchedSequence
    BatchedFrameSequence
    DataFunction
"""

from collections.abc import Callable

import keras
import numpy as np
import tensorflow as tf

from ertk.utils import batch_arrays_by_length, shuffle_multiple

__all__ = [
    "tf_dataset_gen",
    "tf_dataset_mem_ragged",
    "tf_dataset_mem",
    "BatchedSequence",
    "BatchedFrameSequence",
    "DataFunction",
]


DataFunction = Callable[..., tf.data.Dataset]


def tf_dataset_gen(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
):
    """Returns a TensorFlow generator Dataset instance with the given
    data.

    Parameters
    ----------
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

    def gen_inst():
        if shuffle:
            perm = np.random.permutation(len(x))
        else:
            perm = np.arange(len(x))

        if sample_weight is None:
            for i in perm:
                yield x[i], y[i]
        else:
            for i in perm:
                yield x[i], y[i], sample_weight[i]

    sig: tuple[tf.TensorSpec, ...] = (
        tf.TensorSpec(shape=x[0].shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )
    if sample_weight is not None:
        sig += (tf.TensorSpec(shape=(), dtype=tf.float32),)
    data = tf.data.Dataset.from_generator(gen_inst, output_signature=sig)
    return data.batch(batch_size).prefetch(2)


def tf_dataset_mem(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Returns a TensorFlow in-memory Dataset instance with the given
    data.

    Parameters
    ----------
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
    with tf.device("CPU"):
        if sample_weight is None:
            data = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            data = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))

    if shuffle:
        data = data.shuffle(len(x))
    return data.batch(batch_size).prefetch(2)


def tf_dataset_mem_ragged(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Returns a TensorFlow in-memory Dataset instance from
    variable-length features.

    Parameters
    ----------
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
    with tf.device("CPU"):
        if sample_weight is None:
            data = tf.data.Dataset.from_tensor_slices((ragged, y))
        else:
            data = tf.data.Dataset.from_tensor_slices((ragged, y, sample_weight))

    # Group similar lengths in batches, then shuffle batches
    data = data.batch(batch_size)
    if shuffle:
        data = data.shuffle(len(x) // batch_size + 1)

    if sample_weight is None:
        data = data.map(ragged_to_dense)
    else:
        data = data.map(ragged_to_dense_weighted)
    return data.prefetch(2)


class BatchedFrameSequence(keras.utils.Sequence):
    """Creates a sequence of batches of frames to process.

    Parameters
    ----------
    x: ndarray or list of ndarray
        Sequences of vectors.
    y: ndarray
        Labels corresponding to sequences in x.
    prebatched: bool, default = False
        Whether or not x has already been grouped into batches.
    batch_size: int, default = 32
        Batch size to use. Each generated batch will be at most this size.
    shuffle: bool, default = True
        Whether to shuffle the order of the batches.
    """

    def __init__(
        self,
        x: np.ndarray | list[np.ndarray],
        y: np.ndarray,
        prebatched: bool = False,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.x = x
        self.y = y
        if not prebatched:
            self.x, self.y = batch_arrays_by_length(
                self.x, self.y, batch_size=batch_size, shuffle=shuffle
            )
        if shuffle:
            self.x, self.y = shuffle_multiple(self.x, self.y, numpy_indexing=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class BatchedSequence(keras.utils.Sequence):
    """Creates a sequence of batches to process.

    Parameters
    ----------
    x: ndarray or list of ndarray
        Instance feature vectors. Each vector is assumed to be for a different
        instance.
    y: ndarray
        Labels corresponding to sequences in x.
    prebatched: bool, default = False
        Whether or not x has already been grouped into batches.
    batch_size: int, default = 32
        Batch size to use. Each generated batch will be at most this size.
    shuffle: bool, default = True
        Whether to shuffle the instances.
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        if shuffle:
            self.x, self.y = shuffle_multiple(self.x, self.y, numpy_indexing=True)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx: int):
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        return self.x[sl], self.y[sl]


_DATA_FNS = {
    "mem": tf_dataset_mem,
    "gen": tf_dataset_gen,
    "mem_ragged": tf_dataset_mem_ragged,
}


def get_data_fn(name: str) -> Callable[..., tf.data.Dataset]:
    """Returns a data function by name.

    Parameters
    ----------
    name: str
        The name of the data function to return. Can be one of:
        * "``mem``": In-memory data function.
        * "``gen``": Generator data function.
        * "``mem_ragged``": In-memory data function for ragged tensors.

    Returns
    -------
    Callable[..., tf.data.Dataset]
        The data function.
    """
    if ":" in name:
        import importlib

        modname, fname = name.split(":")
        module = importlib.import_module(modname)
        func = getattr(module, fname)
    else:
        func = _DATA_FNS[name]
    return func
