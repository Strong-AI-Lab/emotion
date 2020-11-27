import abc
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import (Any, Callable, Dict, Iterable, Optional, Sequence, Tuple,
                    Union)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import (BaseCrossValidator, KFold,
                                     LeaveOneGroupOut, ParameterGrid)
from sklearn.svm import SVC

from .dataset import CombinedDataset, LabelledDataset
from .utils import shuffle_multiple

__all__ = ['PrecomputedSVC', 'Classifier', 'SKLearnClassifier']

SKClassifierFunction = Callable[[], ClassifierMixin]
ScoreFunction = Callable[[np.ndarray, np.ndarray], float]
KernelFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]

METRICS = ['prec', 'rec', 'uap', 'uar', 'war']


def linear_kernel(x, y) -> np.ndarray:
    return np.matmul(x, y.T)


def poly_kernel(x, y, d=2, r=0,
                gamma: Union[str, float] = 'auto') -> np.ndarray:
    a = np.matmul(x, y.T)
    if gamma == 'auto':
        gamma = 1 / x.shape[1]
    return (gamma * a + r)**d


def rbf_kernel(x, y, gamma: Union[str, float] = 'auto') -> np.ndarray:
    a = np.matmul(x, y.T)
    xx = np.sum(x**2, axis=1)
    yy = np.sum(y**2, axis=1)
    s = xx[:, np.newaxis] + yy[np.newaxis, :]
    if gamma == 'auto':
        gamma = 1 / x.shape[1]
    return np.exp(-gamma * (s - 2 * a))


class PrecomputedSVC(SVC):
    """Class that wraps scikit-learn's SVC to precompute the kernel
    values in order to speed up training. The kernel parameter is a
    string which is transparently mapped to and from the corresponding
    callable function with the relevant parameters (degree, gamma,
    coef0). All other parameters are passed directly to SVC.
    """
    KERNELS = {'rbf': rbf_kernel, 'poly': poly_kernel, 'linear': linear_kernel}

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=1e-3, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', break_ties=False,
                 random_state=None):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties, random_state=random_state
        )
        self.kernel_name = kernel
        self.kernel = self._get_kernel_func()

    def get_params(self, deep) -> Dict[str, Any]:
        params = super().get_params(deep)
        params['kernel'] = self.kernel_name
        return params

    def set_params(self, **params) -> BaseEstimator:
        super().set_params(**params)
        if 'kernel' in params:
            self.kernel_name = params['kernel']
        self.kernel = self._get_kernel_func()
        return self

    def _get_kernel_func(self) -> KernelFunction:
        """Get the kernel function, with parameters, to use in fit() and
        predict(). This is calculated at runtime in order to more easily
        handle changes in parameters such as kernel, gamma, etc.
        """
        f = self.KERNELS[self.kernel]
        params = {}
        if self.kernel == 'poly':
            params = {'d': self.degree, 'r': self.coef0, 'gamma': self.gamma}
        elif self.kernel == 'rbf':
            params = {'gamma': self.gamma}
        return partial(f, **params)


class Classifier(abc.ABC):
    """Base class for classifiers used in test_model()."""

    @abc.abstractmethod
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

    @abc.abstractmethod
    def predict(self, x_test: np.ndarray,
                y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generates predictions for the given input."""
        return NotImplementedError()


class SKLearnClassifier(Classifier):
    """Class wrapper for a scikit-learn classifier instance.

    Parameters:
    -----------
    model_fn: callable
        A callable that returns a new proper classifier that can be
        trained.
    param_grid: dict, optional
    """
    def __init__(self, model_fn: SKClassifierFunction,
                 param_grid: Optional[Dict[str, Sequence]],
                 cv_score_fn: Optional[ScoreFunction]):
        self.model_fn = model_fn
        self.param_grid = ParameterGrid(param_grid)
        self.cv_score_fn = cv_score_fn

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray, fold=None):
        x_train, y_train = shuffle_multiple(x_train, y_train,
                                            numpy_indexing=True)
        x_valid, y_valid = shuffle_multiple(x_valid, y_valid,
                                            numpy_indexing=True)

        if self.param_grid:
            self.clf = optimise_params(
                self.param_grid, self.model_fn, self.cv_score_fn, x_train,
                y_train, x_valid, y_valid, max_workers=24
            )
        else:
            self.clf = self.model_fn()
            self.clf.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray,
                y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.clf.predict(x_test), y_test


def within_corpus_cross_validation(model: Classifier,
                                   dataset: LabelledDataset,
                                   mode: str = 'all',
                                   gender: str = 'all',
                                   reps: int = 1,
                                   splitter: BaseCrossValidator = KFold(10),
                                   validation: str = 'valid'):
    """Cross validates a `Classifier` instance on a single dataset.

    Parameters:
    -----------
    model: Classifier
        The classifier to test.
    dataset: LabelledDataset
        The dataset to test on.
    mode: {'all', 'valence', 'arousal'}
        The kind of classification data to use from the dataset. Default
        is 'all'.
    genders: {'all', 'male', 'female'}
        Which gendered data to perform the test on. Default is 'all'.
    reps: int
        The number of repetitions, default is 1 for a single run.
    splitter: sklearn.model_selection.BaseCrossValidator
        A splitter used for cross-validation. Default is KFold(10) for
        10 fold cross-validation.
    validation: str, {'train', 'valid', 'test'}
        Validation method to use for parameter optimisation. 'train'
        uses training data, 'test' uses test data, 'valid' uses a random
        inner cross-validation fold with the same splitting method.

    Returns:
    --------
    df: pandas.DataFrame
        A dataframe holding the results from all runs with this model.
    """

    if mode == 'all':
        classes = sorted([x[:3] for x in dataset.classes])
    else:
        classes = ['neg', 'pos']

    x = dataset.x
    y = dataset.labels[mode]

    df = pd.DataFrame(
        index=pd.RangeIndex(splitter.get_n_splits(
            x, y, dataset.speaker_indices)),
        columns=pd.MultiIndex.from_product(
            [METRICS, classes, range(reps)], names=['metric', 'class', 'rep'])
    )
    if gender == 'male':
        gender_indices = dataset.male_indices
    elif gender == 'female':
        gender_indices = dataset.female_indices
    else:
        gender_indices = np.arange(len(dataset.names))

    speaker_indices = dataset.speaker_indices[gender_indices]
    groups = dataset.speaker_group_indices[gender_indices]
    x = x[gender_indices]
    y = y[gender_indices]
    for rep in range(reps):
        fold = 0
        # LOSGO cross-validation
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

                    model.fit(x_train, y_train, x_valid, y_valid, fold=fold)
                    # We need to return y_true just in case the order is
                    # modified by batching.
                    y_pred, y_true = model.predict(x_test2, y_test2)
                    _record_metrics(df, fold, rep, y_true, y_pred,
                                    len(classes))
                    fold += 1
            else:
                # TODO: fix this in the general case when using arbitrary
                # cross-validation splitter
                # Make sure we have at least two speakers in the training
                # set so we can use one for validation set.
                if validation == 'valid' and len(
                        np.unique(speaker_indices[train])) >= 2:
                    n_splits = splitter.get_n_splits(
                        x_train, y_train, speaker_indices[train])

                    # Select random inner fold to use as validation set
                    r = np.random.randint(n_splits) + 1
                    splits = splitter.split(x_train, y_train,
                                            speaker_indices[train])
                    for _ in range(r):
                        train2, valid = next(splits)

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
                _record_metrics(df, fold, rep, y_true, y_pred, len(classes))
                fold += 1
    return df


def cross_corpus_cross_validation(clf: Classifier,
                                  combined_dataset: CombinedDataset,
                                  reps: int = 1):
    """Performs cross-validation using each corpus as test set, and the
    rest as training set.

    Args:
    -----
    clf: Classifier
        The classifier to fit and test with the data.
    combined_dataset: CombinedDataset
        The CombinedDataset instance holding the combined data from all
        corpora.
    reps: int
        The number of repetitions to do for each cross-validation round.
    """
    df = pd.DataFrame(
        index=pd.Index(combined_dataset.corpora),
        columns=pd.MultiIndex.from_product(
            [METRICS, combined_dataset.classes, range(reps)],
            names=['metric', 'class', 'rep']
        )
    )
    n_classes = len(combined_dataset.classes)
    for corpus in combined_dataset.corpora:
        print("Fold {}".format(corpus))
        test_idx, train_idx = combined_dataset.get_corpus_split(corpus)
        x_train = combined_dataset.x[train_idx]
        y_train = combined_dataset.y[train_idx]
        x_test = combined_dataset.x[test_idx]
        y_test = combined_dataset.y[test_idx]
        for rep in range(reps):
            print("Rep {}".format(rep))
            clf.fit(x_train, y_train, x_valid=x_train, y_valid=y_train,
                    fold=corpus)
            y_pred, y_true = clf.predict(x_test, y_test)

            # Record metrics
            df.loc[corpus, ('war', slice(None), rep)] = recall_score(
                y_true, y_pred, average='micro')
            df.loc[corpus, ('uar', slice(None), rep)] = recall_score(
                y_true, y_pred, average='macro')
            df.loc[corpus, ('uap', slice(None), rep)] = precision_score(
                y_true, y_pred, average='macro')
            df.loc[corpus, ('rec', slice(None), rep)] = recall_score(
                y_true, y_pred, average=None, labels=list(range(n_classes)))
            df.loc[corpus, ('prec', slice(None), rep)] = precision_score(
                y_true, y_pred, average=None, labels=list(range(n_classes)))
    return df


def test_one_vs_rest(model_fn,
                     dataset: LabelledDataset,
                     gender: str = 'all',
                     reps: int = 1,
                     param_grid: Optional[Dict[str, Any]] = None,
                     splitter: BaseCrossValidator = KFold(10)) -> pd.DataFrame:
    labels = sorted([x[:3] for x in dataset.classes])

    rec = pd.DataFrame(
        index=pd.RangeIndex(splitter.get_n_splits(
            dataset.x, dataset.labels[0], dataset.speaker_indices)),
        columns=pd.MultiIndex.from_product(
            [['prec', 'rec'], labels, list(range(reps))],
            names=['metric', 'class', 'rep']))
    if gender == 'male':
        gender_indices = dataset.male_indices
    elif gender == 'female':
        gender_indices = dataset.female_indices
    else:
        gender_indices = np.arange(len(dataset.names))

    groups = dataset.speaker_indices[gender_indices]
    x = dataset.x[gender_indices]
    for cls in dataset.classes:
        y = dataset.labels[cls][gender_indices]
        for rep in range(reps):
            for fold, (train, test) in enumerate(
                    splitter.split(x, y, groups)):
                x_train, y_train = x[train], y[train]
                x_test, y_test = x[test], y[test]

                if param_grid:
                    classifier = optimise_params(
                        param_grid, model_fn, recall_score, x_train, y_train,
                        x_test, y_test
                    )
                else:
                    classifier = model_fn()

                y_pred = classifier.predict(x_test)
                rec[('prec', cls[:3], rep)][fold] = precision_score(y_test,
                                                                    y_pred)
                rec[('rec', cls[:3], rep)][fold] = recall_score(y_test, y_pred)
    return rec


def _test_one_param(params, cls, score_fn, x_train, y_train, x_valid, y_valid):
    classifier = cls(**params)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_valid)
    score = score_fn(y_valid, y_pred)
    return classifier, score


def optimise_params(param_grid: Iterable[Dict[str, Sequence]],
                    cls: Callable,
                    score_fn: ScoreFunction,
                    x_train: np.ndarray,
                    y_train: np.ndarray,
                    x_valid: np.ndarray,
                    y_valid: np.ndarray,
                    max_workers=len(os.sched_getaffinity(0))) -> BaseEstimator:
    """Performs cross-validation for SKLearnClassifier's using the given
    parameter grid and validation data.

    Returns:
    --------
    classifier
        The best trained classifier for the given parameter
        combinations.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        max_score = -1
        fn = partial(_test_one_param, cls=cls, score_fn=score_fn,
                     x_train=x_train, y_train=y_train, x_valid=x_valid,
                     y_valid=y_valid)
        for clf, score in pool.map(fn, param_grid):
            if score > max_score:
                max_score = score
                classifier = clf
    return classifier


def _record_metrics(df, fold, rep, y_true, y_pred, n_classes):
    df.loc[fold, ('war', slice(None), rep)] = recall_score(y_true, y_pred,
                                                           average='micro')
    df.loc[fold, ('uar', slice(None), rep)] = recall_score(y_true, y_pred,
                                                           average='macro')
    df.loc[fold, ('uap', slice(None), rep)] = precision_score(y_true, y_pred,
                                                              average='macro')
    df.loc[fold, ('rec', slice(None), rep)] = recall_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)))
    df.loc[fold, ('prec', slice(None), rep)] = precision_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)))


def print_results(df: pd.DataFrame):
    """Prints the results dataframe in a nice format."""
    metrics = df.axes[1].get_level_values('metric').unique()
    labels = df.axes[1].get_level_values('class').unique()
    print()
    print("Metrics: mean +- std. dev. over folds")
    print("Across reps:")
    print('           ' + ' '.join(['{:<12}'.format(c) for c in labels]))
    for metric in metrics:
        print('{:<4s} {}'.format(metric, ' '.join([
            '{:<4.2f} +- {:<4.2f}'.format(df[(metric, c)].mean().mean(),
                                          df[(metric, c)].std().mean())
            for c in labels
        ])))
    print()
    print("Across classes and reps:")
    for metric in metrics:
        print('{:<4s}: {:.3f} +- {:.2f} ({:.2f})'.format(
            metric.upper(), df[metric].mean().mean(), df[metric].std().mean(),
            df[metric].max().max()
        ))
    print("")
    print()
