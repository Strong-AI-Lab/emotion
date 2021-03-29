import abc
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import BaseCrossValidator, KFold, LeaveOneGroupOut

from .dataset import CombinedDataset, LabelledDataset

ScoreFunction = Callable[[np.ndarray, np.ndarray], float]

METRICS = ["prec", "rec", "uap", "uar", "war"]


class Classifier(abc.ABC):
    """Base class for classifiers used in test_model()."""

    @abc.abstractmethod
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        fold: Optional[Union[int, str]] = None,
    ):
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
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates predictions for the given input."""
        raise NotImplementedError()


def within_corpus_cross_validation(
    model: Classifier,
    x: np.ndarray,
    y: np.ndarray,
    speakers: np.ndarray,
    groups: np.ndarray,
    classes: List[str],
    reps: int = 1,
    splitter: BaseCrossValidator = KFold(10),
    validation: str = "valid",
):
    """Cross validates a `Classifier` instance on a single dataset.

    Parameters:
    -----------
    model: Classifier
        The classifier to test.
    x: numpy.ndarray
        Full feature matrix
    y: numpy.ndarray
        Labels for x.
    speakers: numpy.ndarray
        Speakers for x.
    groups: numpy.ndarray
        Speaker groups for x.
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
    folds = splitter.get_n_splits(x, y, speakers)
    df = pd.DataFrame(
        index=pd.RangeIndex(1, folds + 1),
        columns=pd.MultiIndex.from_product(
            [METRICS, classes, range(reps)], names=["metric", "class", "rep"]
        ),
    )

    fold = 1
    # LOSGO cross-validation
    for train, test in splitter.split(x, y, groups):
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        # This checks to see if the test set still has different
        # speakers, so that we can validate using each of them. This is
        # used for IEMOCAP and MSP-IMPROV sessions.
        n_splits = splitter.get_n_splits(x_test, y_test, speakers[test])
        if n_splits > 1 and isinstance(splitter, LeaveOneGroupOut):
            for valid, test in splitter.split(x_test, y_test, speakers[test]):
                print(f"Fold {fold}/{folds}")

                x_valid = x_test[valid]
                y_valid = y_test[valid]
                x_test2 = x_test[test]
                y_test2 = y_test[test]

                model.fit(x_train, y_train, x_valid, y_valid, fold=fold)
                # We need to return y_true just in case the order is
                # modified by batching.
                y_pred, y_true = model.predict(x_test2, y_test2)
                _record_metrics(df, fold, y_true, y_pred, len(classes))
                fold += 1
        else:
            # TODO: fix this in the general case when using arbitrary
            # cross-validation splitter
            # Make sure we have at least two speakers in the training
            # set so we can use one for validation set.
            if validation == "valid" and len(np.unique(speakers[train])) >= 2:
                n_splits = splitter.get_n_splits(x_train, y_train, speakers[train])

                # Select random inner fold to use as validation set
                r = np.random.default_rng().integers(n_splits) + 1
                splits = splitter.split(x_train, y_train, speakers[train])
                for _ in range(r):
                    train2, valid = next(splits)

                x_valid = x_train[valid]
                y_valid = y_train[valid]
                x_train = x_train[train2]
                y_train = y_train[train2]
            elif validation == "test":
                x_valid = x_test
                y_valid = y_test
            else:
                x_valid = x_train
                y_valid = y_train

            print(f"Fold {fold}/{folds}")
            model.fit(x_train, y_train, x_valid, y_valid, fold=fold)
            y_pred, y_true = model.predict(x_test, y_test)
            _record_metrics(df, fold, y_true, y_pred, len(classes))
            fold += 1
    return df


def cross_corpus_cross_validation(
    clf: Classifier, combined_dataset: CombinedDataset, reps: int = 1
):
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
            names=["metric", "class", "rep"],
        ),
    )
    n_classes = len(combined_dataset.classes)
    for corpus in combined_dataset.corpora:
        print(f"Fold {corpus}")
        test_idx, train_idx = combined_dataset.get_corpus_split(corpus)
        x_train = combined_dataset.x[train_idx]
        y_train = combined_dataset.y[train_idx]
        x_test = combined_dataset.x[test_idx]
        y_test = combined_dataset.y[test_idx]
        for rep in range(reps):
            print(f"Rep {rep}")
            clf.fit(x_train, y_train, x_valid=x_train, y_valid=y_train, fold=corpus)
            y_pred, y_true = clf.predict(x_test, y_test)

            # Record metrics
            df.loc[corpus, ("war", slice(None), rep)] = recall_score(
                y_true, y_pred, average="micro"
            )
            df.loc[corpus, ("uar", slice(None), rep)] = recall_score(
                y_true, y_pred, average="macro"
            )
            df.loc[corpus, ("uap", slice(None), rep)] = precision_score(
                y_true, y_pred, average="macro"
            )
            df.loc[corpus, ("rec", slice(None), rep)] = recall_score(
                y_true, y_pred, average=None, labels=list(range(n_classes))
            )
            df.loc[corpus, ("prec", slice(None), rep)] = precision_score(
                y_true, y_pred, average=None, labels=list(range(n_classes))
            )
    return df


def test_one_vs_rest(
    model_fn,
    dataset: LabelledDataset,
    gender: str = "all",
    reps: int = 1,
    param_grid: Optional[Dict[str, Any]] = None,
    splitter: BaseCrossValidator = KFold(10),
) -> pd.DataFrame:
    labels = sorted([x[:3] for x in dataset.classes])

    rec = pd.DataFrame(
        index=pd.RangeIndex(
            splitter.get_n_splits(dataset.x, dataset.y, dataset.speaker_indices)
        ),
        columns=pd.MultiIndex.from_product(
            [["prec", "rec"], labels, list(range(reps))],
            names=["metric", "class", "rep"],
        ),
    )
    if gender == "male":
        gender_indices = dataset.male_indices
    elif gender == "female":
        gender_indices = dataset.female_indices
    else:
        gender_indices = np.arange(len(dataset.names))

    groups = dataset.speaker_indices[gender_indices]
    x = dataset.x[gender_indices]
    for cls in dataset.classes:
        y = dataset.labels[cls][gender_indices]
        for rep in range(reps):
            for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
                x_train, y_train = x[train], y[train]
                x_test, y_test = x[test], y[test]

                if param_grid:
                    classifier = optimise_params(
                        param_grid,
                        model_fn,
                        recall_score,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                    )
                else:
                    classifier = model_fn()

                y_pred = classifier.predict(x_test)
                rec[("prec", cls[:3], rep)][fold] = precision_score(y_test, y_pred)
                rec[("rec", cls[:3], rep)][fold] = recall_score(y_test, y_pred)
    return rec


def _test_one_param(params, cls, score_fn, x_train, y_train, x_valid, y_valid):
    classifier = cls(**params)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_valid)
    score = score_fn(y_valid, y_pred)
    return classifier, score


def optimise_params(
    param_grid: Iterable[Dict[str, Sequence]],
    cls: Callable,
    score_fn: ScoreFunction,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    max_workers=None,
) -> BaseEstimator:
    """Performs cross-validation using the given parameter grid and
    validation data.

    Returns:
    --------
    classifier
        The best trained classifier for the given parameter
        combinations.
    """
    if max_workers is None:
        try:
            max_workers = len(os.sched_getaffinity(0))
        except AttributeError:  # sched_getaffinity is only on Unix
            max_workers = os.cpu_count()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        max_score = -1
        fn = partial(
            _test_one_param,
            cls=cls,
            score_fn=score_fn,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
        )
        for clf, score in pool.map(fn, param_grid):
            if score > max_score:
                max_score = score
                classifier = clf
    return classifier


def _record_metrics(df, fold, y_true, y_pred, n_classes):
    df.loc[fold, ("war", slice(None))] = recall_score(y_true, y_pred, average="micro")
    df.loc[fold, ("uar", slice(None))] = recall_score(y_true, y_pred, average="macro")
    df.loc[fold, ("uap", slice(None))] = precision_score(
        y_true, y_pred, average="macro"
    )
    df.loc[fold, ("rec", slice(None))] = recall_score(
        y_true, y_pred, average=None, labels=list(range(n_classes))
    )
    df.loc[fold, ("prec", slice(None))] = precision_score(
        y_true, y_pred, average=None, labels=list(range(n_classes))
    )


def print_results(df: pd.DataFrame):
    """Prints the results dataframe in a nice format."""
    metrics = df.axes[1].get_level_values("metric").unique()
    labels = df.axes[1].get_level_values("class").unique()
    print()
    print("Metrics: mean +- std. dev. over folds")
    print("Across reps:")
    print("           " + " ".join([f"{c:<12}" for c in labels]))
    for metric in metrics:
        class_scores = [
            f"{df[(metric, c)].mean().mean():<4.2f} +- "
            f"{df[(metric, c)].mean().std():<4.2f}"
            for c in labels
        ]
        print(f'{metric:<4s} {" ".join(class_scores)}')
    print()
    print("Across classes and reps:")
    for metric in metrics:
        print(
            f"{metric.upper():<4s}: {df[metric].mean().mean():.3f} +- "
            f"{df[metric].mean().std():.2f} ({df[metric].max().max():.2f})"
        )
    print("")
    print()
