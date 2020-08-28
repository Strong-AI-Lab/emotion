import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import (BaseCrossValidator, KFold,
                                     LeaveOneGroupOut, ParameterGrid)
from sklearn.svm import SVC
from tqdm.keras import TqdmCallback

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer

from .dataset import LabelledDataset
from .tensorflow.classification import tf_classification_metrics
from .utils import shuffle_multiple

ModelFunction = Callable[[], Union[ClassifierMixin, Model]]
DataFunction = Callable[[np.ndarray, np.ndarray], tf.data.Dataset]
ScoreFunction = Callable[[np.ndarray, np.ndarray], float]

METRICS = ['prec', 'rec', 'uap', 'uar', 'war']


def linear_kernel(x, y):
    return np.matmul(x, y.T)


def poly_kernel(x, y, d=2, r=0, gamma='auto'):
    M = linear_kernel(x, y)
    if gamma == 'auto':
        gamma = 1 / x.shape[1]
    return (gamma * M + r)**d


def rbf_kernel(x, y, gamma='auto'):
    M = linear_kernel(x, y)
    xx = np.sum(x**2, axis=1)
    yy = np.sum(y**2, axis=1)
    D = xx[:, np.newaxis] + yy[np.newaxis, :]
    if gamma == 'auto':
        gamma = 1 / x.shape[1]
    return np.exp(-gamma * (D - 2 * M))


class PrecomputedSVC:
    def __init__(self, C=1, kernel='rbf', degree=3, gamma='auto', coef0=0,
                 **clf_kwargs):
        if kernel == 'linear':
            self.kernel_func = linear_kernel
        elif kernel == 'poly':
            self.kernel_func = partial(poly_kernel, d=degree, r=coef0,
                                       gamma=gamma)
        elif kernel == 'rbf':
            self.kernel_func = partial(rbf_kernel, gamma=gamma)
        else:
            raise ValueError(
                "kernel must be in {{'linear', 'poly', 'rbf'}}, got '{}'"
                .format(kernel))
        self.clf = SVC(C=C, kernel='precomputed', **clf_kwargs)

    def fit(self, x_train, y_train, sample_weight=None,
            **kwargs):
        self.x_train = x_train
        K = self.kernel_func(x_train, x_train)
        return self.clf.fit(K, y_train, sample_weight=sample_weight, **kwargs)

    def predict(self, x_test, **kwargs):
        K = self.kernel_func(x_test, self.x_train)
        return self.clf.predict(K)


class Classifier:
    """Base class for classifiers used in test_model().

    Parameters:
    -----------
    model_fn: callable
        A callable that returns a new proper classifier that can be trained.
    """

    def __init__(self, model_fn: ModelFunction):
        self.model_fn = model_fn

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray,
            fold: Optional[int] = None):
        """Fits this classifier to the training data.

        Parameters:
        -----------
        x_train, y_train: numpy.ndarray
            Training data.
        x_test, y_test: numpy.ndarray
            Testing data.
        fold: int, optional, default = 0
            The current fold, for logging purposes.
        """
        return NotImplementedError()

    def predict(self, x_test: np.ndarray,
                y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generates predictions for the given input."""
        return NotImplementedError()


class SKLearnClassifier(Classifier):
    """Class wrapper for a scikit-learn classifier instance.

    Parameters:
    -----------
    model_fn: callable
        A callable that returns a new proper classifier that can be trained.
    param_grid: dict, optional
    """
    def __init__(self, model_fn: ModelFunction,
                 param_grid: Optional[Dict[str, Sequence]],
                 cv_score_fn: Optional[ScoreFunction]):
        super().__init__(model_fn)
        self.param_grid = ParameterGrid(param_grid)
        self.cv_score_fn = cv_score_fn

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray, fold=None):
        x_train, y_train = shuffle_multiple(x_train, y_train,
                                            numpy_indexing=True)
        x_valid, y_valid = shuffle_multiple(x_valid, y_valid,
                                            numpy_indexing=True)

        if self.param_grid:
            self.clf = cross_validate(
                self.param_grid, self.model_fn, self.cv_score_fn, x_train,
                y_train, x_valid, y_valid
            )
        else:
            self.clf = self.model_fn()
            self.clf.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray,
                y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.clf.predict(x_test), y_test


class TFClassifier(Classifier):
    """Class wrapper for a TensorFlow Keras classifier model.

    Parameters:
    -----------
    model_fn: callable
        A callable that returns a new proper classifier that can be trained.
    n_epochs: int, optional, default = 50
        Maximum number of epochs to train for.
    class_weight: dict, optional
        A dictionary mapping class IDs to weights. Default is to ignore
        class weights.
    data_fn: callable, optional
        Callable that takes x and y as input and returns a
        tensorflow.keras.Sequence object or a tensorflow.data.Dataset
        object.
    callbacks: list, optional
        A list of tensorflow.keras.callbacks.Callback objects to use during
        training. Default is an empty list, so that the default Keras
        callbacks are used.
    loss: keras.losses.Loss
        The loss to use. Default is
        tensorflow.keras.losses.SparseCategoricalCrossentropy.
    optimizer: keras.optimizers.Optimizer
        The optimizer to use. Default is tensorflow.keras.optimizers.Adam.
    verbose: bool, default = False
        Whether to output details per epoch.
    """
    def __init__(self, model_fn: ModelFunction,
                 n_epochs: int = 50,
                 class_weight: Optional[Dict[int, float]] = None,
                 data_fn: Optional[DataFunction] = None,
                 callbacks: List[Callback] = [],
                 loss: Loss = SparseCategoricalCrossentropy(),
                 optimizer: Optimizer = Adam(),
                 verbose: bool = False):
        super().__init__(model_fn)
        self.n_epochs = n_epochs
        self.class_weight = class_weight
        if data_fn is not None:
            self.data_fn = data_fn
        self.callbacks = callbacks
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose

    def data_fn(self, x: np.ndarray, y: np.ndarray,
                shuffle: bool = True) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            dataset = dataset.shuffle(len(x))
        return dataset

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray, fold: int = 0):
        # Clear graph
        keras.backend.clear_session()
        # Reset optimiser and loss
        optimizer = self.optimizer.from_config(self.optimizer.get_config())
        loss = self.loss.from_config(self.loss.get_config())
        for cb in self.callbacks:
            if isinstance(cb, TensorBoard):
                cb.log_dir = str(Path(cb.log_dir).parent / str(fold))
        callbacks = self.callbacks + [
            TqdmCallback(epochs=self.n_epochs, verbose=0)]

        self.model = self.model_fn()
        self.model.compile(loss=loss, optimizer=optimizer,
                           metrics=tf_classification_metrics())

        train_data = self.data_fn(x_train, y_train, shuffle=True)
        valid_data = self.data_fn(x_valid, y_valid, shuffle=True)
        self.model.fit(
            train_data,
            epochs=self.n_epochs,
            class_weight=self.class_weight,
            validation_data=valid_data,
            callbacks=callbacks,
            verbose=int(self.verbose)
        )

    def predict(self, x_test: np.ndarray,
                y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        test_data = self.data_fn(x_test, y_test, shuffle=False)
        y_true = np.concatenate([y for _, y in test_data])
        return np.argmax(self.model.predict(test_data), axis=1), y_true


def test_model(model: Classifier,
               dataset: LabelledDataset,
               mode: str = 'all',
               genders: List[str] = ['all'],
               reps: int = 1,
               splitter: BaseCrossValidator = KFold(10),
               validation: str = 'valid'):
    """Tests a `Classifier` instance on the given dataset.

    Parameters:
    -----------
    model: Classifier
        The classifier to test.
    dataset: LabelledDataset
        The dataset to test on.
    mode: {'all', 'valence', 'arousal'}
        The kind of classification data to use from the dataset.
    genders: list
        Which gendered data to perform the test on: 'm' for male, 'f' for
        female, and 'all' for combined. Default is ['all'].
    reps: int
        The number of repetitions, default is 1 for a single run.
    splitter: sklearn.model_selection.BaseCrossValidator
        A splitter used for cross-validation. Default is KFold(10) for 10 fold
        cross-validation.
    validation: str, {'train', 'valid', 'test'}
        Validation method to use for parameter optimisation. 'train' uses
        training data, 'test' uses test data, 'valid' uses a random inner
        cross-validation fold with the same splitting method.

    Returns:
    --------
    df: pandas.DataFrame
        A dataframe holding the results from all runs with this model.
    """

    if mode == 'all':
        labels = sorted([x[:3] for x in dataset.classes])
    else:
        labels = ['neg', 'pos']

    df = pd.DataFrame(
        index=pd.RangeIndex(splitter.get_n_splits(
            dataset.x, dataset.labels[mode], dataset.speaker_indices)),
        columns=pd.MultiIndex.from_product(
            [METRICS, genders, labels, range(reps)],
            names=['metric', 'gender', 'class', 'rep']
        )
    )
    for gender in genders:
        gender_indices = dataset.gender_indices[gender]
        speaker_indices = dataset.speaker_indices[gender_indices]
        groups = dataset.speaker_group_indices[gender_indices]
        x = dataset.x[gender_indices]
        y = dataset.labels[mode][gender_indices]
        for rep in range(reps):
            fold = 0
            for train, test in splitter.split(x, y, groups):
                x_train = x[train]
                y_train = y[train]
                x_test = x[test]
                y_test = y[test]

                # This checks to see if the test set still has different
                # speakers, so that we can validate using each of them.
                n_splits = splitter.get_n_splits(x_test, y_test,
                                                 speaker_indices[test])
                if n_splits > 1 and isinstance(splitter, LeaveOneGroupOut):
                    for valid, test in splitter.split(x_test, y_test,
                                                      speaker_indices[test]):
                        print("Fold {}".format(fold + 1))

                        x_valid = x_test[valid]
                        y_valid = y_test[valid]
                        x_test2 = x_test[test]
                        y_test2 = y_test[test]

                        model.fit(x_train, y_train, x_valid, y_valid,
                                  fold=fold)
                        # We need to return y_true just in case the order is
                        # modified by batching.
                        y_pred, y_true = model.predict(x_test2, y_test2)
                        _record_metrics(df, fold, rep, y_true, y_pred,
                                       len(labels))
                        fold += 1
                else:
                    # TODO: fix this in the general case when using arbitrary
                    # cross-validation splitter
                    if validation == 'valid' and len(
                            np.unique(speaker_indices[train])) >= 2:
                        n_splits = splitter.get_n_splits(
                            x_train, y_train, speaker_indices[train])

                        # Select random inner fold to use as validation set
                        r = np.random.randint(n_splits) + 1
                        for _ in range(r):
                            train2, valid = next(splitter.split(
                                x_train, y_train, speaker_indices[train]))
                        x_valid = x_train[valid]
                        y_valid = y_train[valid]
                        x_train = x_train[train2]
                        y_train = y_train[train2]
                    elif validation == 'test':
                        x_valid = x_test
                        y_valid = y_test
                    else:
                        x_valid = x_train
                        y_valid = y_train

                    print("Fold {}".format(fold + 1))
                    model.fit(x_train, y_train, x_valid, y_valid, fold=fold)
                    y_pred, y_true = model.predict(x_test, y_test)
                    _record_metrics(df, fold, rep, y_true, y_pred, len(labels))
                    fold += 1
    return df


def test_one_vs_rest(model_fn,
                     dataset: LabelledDataset,
                     gendered=False,
                     reps=1,
                     param_grid=None,
                     splitter=KFold(10)) -> pd.DataFrame:
    genders = ['all', 'f', 'm'] if gendered else ['all']
    labels = sorted([x[:3] for x in dataset.classes])

    rec = pd.DataFrame(
        index=pd.RangeIndex(splitter.get_n_splits(
            dataset.x, dataset.labels[0], dataset.speaker_indices)),
        columns=pd.MultiIndex.from_product(
            [['prec', 'rec'], genders, labels, list(range(reps))],
            names=['metric', 'gender', 'class', 'rep']))
    for gender in genders:
        groups = dataset.speaker_indices[dataset.gender_indices[gender]]
        x = dataset.x[dataset.gender_indices[gender]]
        for idx in range(dataset.n_classes):
            y = dataset.labels[idx][dataset.gender_indices[gender]]
            for rep in range(reps):
                for fold, (train, test) in enumerate(
                        splitter.split(x, y, groups)):
                    x_train, y_train = x[train], y[train]
                    x_test, y_test = x[test], y[test]

                    if param_grid:
                        classifier = cross_validate(param_grid, model_fn,
                                                    recall_score, x_train,
                                                    y_train, x_test, y_test)
                    else:
                        classifier = model_fn()

                    y_pred = classifier.predict(x_test)
                    rec[('prec', gender, labels[idx], rep)][fold] \
                        = precision_score(y_test, y_pred)
                    rec[('rec', gender, labels[idx], rep)][fold] \
                        = recall_score(y_test, y_pred)
    return rec


def _test_one_param(params, klass, score_fn, x_train, y_train, x_valid,
                    y_valid):
    classifier = klass(**params)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_valid)
    score = score_fn(y_valid, y_pred)
    return classifier, score


def cross_validate(param_grid, klass, score_fn, x_train, y_train, x_valid,
                   y_valid):
    """Performs cross-validation for SKLearnClassifier's using the given
    parameter grid and validation data.

    Returns:
    --------
    classifier
        The best trained classifier for the given parameter combinations.
    """
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        max_score = -1
        fn = partial(_test_one_param, klass=klass, score_fn=score_fn,
                     x_train=x_train, y_train=y_train, x_valid=x_valid,
                     y_valid=y_valid)
        for clf, score in pool.map(fn, param_grid):
            if score > max_score:
                max_score = score
                classifier = clf
    return classifier


def _record_metrics(df, fold, rep, y_true, y_pred, n_classes):
    df.loc[fold, ('war', 'all', slice(None), rep)] = recall_score(
        y_true, y_pred, average='micro')
    df.loc[fold, ('uar', 'all', slice(None), rep)] = recall_score(
        y_true, y_pred, average='macro')
    df.loc[fold, ('uap', 'all', slice(None), rep)] = precision_score(
        y_true, y_pred, average='macro')
    df.loc[fold, ('rec', 'all', slice(None), rep)] = recall_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)))
    df.loc[fold, ('prec', 'all', slice(None), rep)] = precision_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)))


def print_results(df: pd.DataFrame):
    """Prints the results dataframe in a nice format."""
    genders = df.axes[1].get_level_values('gender').unique()
    metrics = df.axes[1].get_level_values('metric').unique()
    labels = df.axes[1].get_level_values('class').unique()
    for gender in genders:
        print()
        print("Metrics: mean +- std. dev. over folds")
        print("Across reps:")
        print('           ' + ' '.join(
            ['{:<12}'.format(c) for c in labels]))
        for metric in metrics:
            print('{:<4s} ({:^3s}) {}'.format(metric, gender, ' '.join(
                ['{:<4.2f} +- {:<4.2f}'.format(
                  df[(metric, gender, c)].mean().mean(),
                  df[(metric, gender, c)].std().mean()) for c in labels])))
        print()
        print("Across classes and reps:")
        for metric in metrics:
            print('{:<4s}: {:.3f} +- {:.2f} ({:.2f})'.format(
                metric.upper(),
                df[(metric, gender)].mean().mean(),
                df[(metric, gender)].std().mean(),
                df[(metric, gender)].max().max()))
        print("")
        print()
