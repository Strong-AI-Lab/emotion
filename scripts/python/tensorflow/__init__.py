from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow import keras

from ..dataset import ArffDataset, FrameDataset
from ..classification import record_metrics, METRICS

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
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_true = tf.squeeze(y_true, axis=[-1])
        y_true_int = tf.cast(y_true, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)


class BatchedFrameSequence(keras.utils.Sequence):
    def __init__(self, x, y, prebatched=False):
        self.x = x
        self.y = y
        if not prebatched:
            self.x, self.y = FrameDataset.batch_arrays(self.x, self.y)
        perm = np.random.permutation(len(self.x))
        self.x = [self.x[i] for i in perm]
        self.y = [self.y[i] for i in perm]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BatchedSequence(keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
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


def plot_training(trainings, dataset):
    n_epochs = max(len(t.epoch) for t in trainings) - 1
    max_loss = np.max([np.max(t.history['loss']) for t in trainings])
    n_instances = dataset.n_instances
    n_classes = dataset.n_classes

    plt.figure()
    for training in trainings:
        plt.plot(training.history['loss'], color='red')
        plt.plot(training.history['val_loss'], color='blue')
    plt.axis([0, n_epochs, 0, max_loss])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.figure()
    for training in trainings:
        plt.plot(training.history['war'], color='red')
        plt.plot(training.history['val_war'], color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('WAR')
    max_class = np.max(np.bincount(dataset.y.astype(int)))
    plt.plot([0, n_epochs], [max_class / n_instances, max_class / n_instances])
    plt.axis([0, n_epochs, 0, 1])

    plt.figure()
    for training in trainings:
        plt.plot(training.history['uar'], color='red')
        plt.plot(training.history['val_uar'], color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('UAR')
    plt.plot([0, n_epochs], [1 / n_classes, 1 / n_classes])
    plt.axis([0, n_epochs, 0, 1])

    plt.show()


def get_tf_dataset(x, y, batch_size=32):
    if isinstance(x, list):
        return BatchedFrameSequence(x, y)
    return BatchedSequence(x, y, batch_size)


def test_model(model_fn,
               dataset: ArffDataset,
               mode='all',
               gendered=False,
               reps=1,
               splitter=KFold(10),
               class_weight=None,
               callback_fn=None,
               data_fn=None,
               batch_size=50,
               n_epochs=50):

    if mode == 'all':
        labels = sorted([x[:3] for x in dataset.classes])
    else:
        labels = ['neg', 'pos']

    genders = ['all', 'f', 'm'] if gendered else ['all']

    if class_weight == 'balanced':
        class_weight = ((dataset.n_instances / dataset.n_classes)
                        / np.bincount(dataset.y.astype(np.int)))
        class_weight = dict(zip(range(dataset.n_classes), class_weight))
    if data_fn is None:
        data_fn = partial(get_tf_dataset, batch_size=batch_size)
    if callback_fn is None:
        callback_fn = list

    df = pd.DataFrame(
        index=pd.RangeIndex(splitter.get_n_splits(
            dataset.x, dataset.labels[mode], dataset.speaker_indices)),
        columns=pd.MultiIndex.from_product(
            [METRICS, genders, labels, range(reps)],
            names=['metric', 'gender', 'class', 'rep'])
    )

    trainings = []
    for gender in genders:
        groups = dataset.speaker_indices[dataset.gender_indices[gender]]
        if isinstance(dataset.x, list):
            x = [dataset.x[i] for i in dataset.gender_indices[gender]]
        else:
            x = dataset.x[dataset.gender_indices[gender]]
        y = dataset.labels[mode]
        for rep in range(reps):
            for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
                model = model_fn()

                if isinstance(x, list):
                    x_train = [x[i] for i in train]
                    x_test = [x[i] for i in test]
                else:
                    x_train = x[train]
                    x_test = x[test]
                y_train = y[train]
                y_test = y[test]

                train_data = data_fn(x_train, y_train)
                test_data = data_fn(x_test, y_test)

                training = model.fit(
                    train_data,
                    epochs=n_epochs,
                    class_weight=class_weight,
                    validation_data=test_data,
                    callbacks=callback_fn(),
                    verbose=0
                )
                trainings.append(training)

                y_pred = np.argmax(model.predict(test_data), axis=1)
                y_test = np.concatenate([x[1] for x in test_data])

                record_metrics(df, fold, rep, y_test, y_pred, len(labels))
    return df, trainings
