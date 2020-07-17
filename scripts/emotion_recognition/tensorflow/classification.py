from typing import List, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.utils import Sequence

from emotion_recognition.utils import shuffle_multiple


def batch_arrays(arrays_x: List[np.ndarray], y: np.ndarray,
                 batch_size: int = 32, shuffle: bool = True,
                 uniform_batch_size: bool = False):
    """Batches a list of arrays of different sizes, grouping them by size. This
    is designed for use with variable length sequences.

    Parameters:
    -----
    arrays_x: list of ndarray
        A list of arrays, possibly of different sizes, to batch.
    y: ndarray
        The labels for each of the arrays in arrays_x.
    batch_size: int
        Arrays will be grouped together by size, up to a maximum of batch_size,
        after which a new batch will be created. Thus each batch produced will
        have between 1 and batch_size items.
    shuffle: bool, default = True
        Whether to shuffle array order in a batch.
    uniform_batch_size: bool, default = False
        Whether to keep all batches the same size, batch_size, and pad with
        zeros if necessary, or have batches of different sizes if there aren't
        enough sequences to group together.

    Returns:
    --------
    x_list: ndarray,
        The batched arrays. x_list[i] is the i'th
        batch, having between 1 and batch_size items, each of length
        lengths[i].
    y_list: ndarray
        The batched labels corresponding to sequences in x_list. y_list[i] has
        the same size as x_list[i].
    """
    if shuffle:
        arrays_x, y = shuffle_multiple(arrays_x, y, numpy_indexing=False)

    sizes = [x.shape[0] for x in arrays_x]
    lengths, indices = np.unique(sizes, return_inverse=True)

    x_list = []
    y_list = []
    for l_idx in range(len(lengths)):
        idx = np.arange(len(arrays_x))[indices == l_idx]
        for b in range(0, len(idx), batch_size):
            b_idx = idx[b:b + batch_size]
            arr = np.zeros((len(b_idx), lengths[l_idx], arrays_x[0].shape[1]),
                           dtype=np.float32)
            for k, j in enumerate(b_idx):
                arr[k, :, :] = arrays_x[j]
            x_list.append(arr)
            y_list.append(y[b_idx])
    x_list = np.array(x_list, dtype=object)
    y_list = np.array(y_list, dtype=object)
    return x_list, y_list


class BalancedSparseCategoricalAccuracy(SparseCategoricalAccuracy):
    """Calculates categorical accuracy with class weights inversely
    proportional to their size. This behaves as if classes are balanced having
    the same number of instances, and is equivalent to the arithmetic mean
    recall over all classes.
    """
    def __init__(self, name='balanced_sparse_categorical_accuracy', **kwargs):
        super().__init__(name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)


class BatchedFrameSequence(Sequence):
    """Creates a sequence of batches of frames to process.

    Parameters:
    -----------
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
    def __init__(self, x: Union[np.ndarray, List[np.ndarray]],
                 y: np.ndarray, prebatched: bool = False, batch_size: int = 32,
                 shuffle: bool = True):
        self.x = x
        self.y = y
        if not prebatched:
            self.x, self.y = batch_arrays(
                self.x, self.y, batch_size=batch_size, shuffle=shuffle)
        if shuffle:
            self.x, self.y = shuffle_multiple(self.x, self.y,
                                              numpy_indexing=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class BatchedSequence(Sequence):
    """Creates a sequence of batches to process.

    Parameters:
    -----------
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
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32,
                 shuffle: bool = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        if shuffle:
            self.x, self.y = shuffle_multiple(self.x, self.y,
                                              numpy_indexing=True)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx: int):
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        return self.x[sl], self.y[sl]


def tf_classification_metrics():
    return [SparseCategoricalAccuracy(name='war'),
            BalancedSparseCategoricalAccuracy(name='uar')]
