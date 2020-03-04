import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..dataset import FrameDataset

__all__ = [
    'BalancedSparseCategoricalAccuracy',
    'BatchedSequence',
    'BatchedFrameSequence',
    'plot_training',
    'test_model',
    'tf_classification_metrics'
]


class BalancedSparseCategoricalAccuracy(
        keras.metrics.SparseCategoricalAccuracy):
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


class BatchedFrameSequence(keras.utils.Sequence):
    def __init__(self, x, y, prebatched=False, batch_size=32, shuffle=True):
        self.x = x
        self.y = y
        if not prebatched:
            self.x, self.y = FrameDataset.batch_arrays(self.x, self.y,
                                                       batch_size=batch_size)
        if shuffle:
            perm = np.random.permutation(len(self.x))
            self.x = self.x[perm]
            self.y = self.y[perm]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BatchedSequence(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=32, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        if shuffle:
            perm = np.random.permutation(len(self.x))
            self.x = self.x[perm]
            self.y = self.y[perm]

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        return self.x[sl], self.y[sl]


def tf_classification_metrics():
    return [
        keras.metrics.SparseCategoricalAccuracy(name='war'),
        BalancedSparseCategoricalAccuracy(name='uar')
    ]
