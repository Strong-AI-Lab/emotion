"""
Classification functions.

This module contains functions for performing classification tasks.

.. autosummary::
    :toctree: generated/

    standard_class_scoring
    cross_validate
    dataset_cross_validation
    dataset_train_val_test
    get_balanced_class_weights
    get_balanced_sample_weights
    class_ratings_to_probs
"""

import logging
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    get_scorer,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import BaseCrossValidator, LeaveOneGroupOut

from ertk.dataset import Dataset
from ertk.metrics import binary_accuracy_score, make_scoring_dict
from ertk.train import ExperimentResult, get_cv_splitter, scores_to_df
from ertk.utils import ScoreFunction

__all__ = [
    "standard_class_scoring",
    "cross_validate",
    "dataset_cross_validation",
    "dataset_train_val_test",
    "get_balanced_class_weights",
    "get_balanced_sample_weights",
    "class_ratings_to_probs",
]


def standard_class_scoring(classes: Sequence[str]) -> dict[str, Callable]:
    """Given a list of classes, returns scikit-learn scorers for overall
    metrics and per-class metrics, for multiclass classification.

    The metrics are:
        Overall:
            uar: Balanced accuracy
            war: Accuracy
            microf1: Micro averaged F1 score
            macrof1: Macro averaged F1 score
        Per-class:
            {class}_rec: Recall
            {class}_prec: Precision
            {class}_f1: F1 score
            {class}_ba: Binary accuracy

    Binary accuracy is calculated as the accuracy when considering
    labels for a class in a one-vs-rest scenario (i.e. the sum of cell
    (i, i) in the confusion matrix and all other cells not in row or
    column i).
    """

    # FIXME: Use labels param with macro and micro recall instead of
    # (balanced) accuracy in order to account for absent classes from
    # test data? Or rely on user to create valid train/test splits.
    scoring = {
        "uar": get_scorer("balanced_accuracy"),
        "war": get_scorer("accuracy"),
        "microf1": make_scorer(f1_score, average="micro"),
        "macrof1": make_scorer(f1_score, average="macro"),
    }
    for i, c in enumerate(classes):
        scoring.update(
            {
                f"{c}_rec": make_scorer(
                    recall_score, average=None, labels=[i], zero_division=0
                ),
                f"{c}_prec": make_scorer(
                    precision_score, average=None, labels=[i], zero_division=0
                ),
                f"{c}_f1": make_scorer(
                    f1_score, average=None, labels=[i], zero_division=0
                ),
                f"{c}_ba": make_scorer(binary_accuracy_score, average=None, labels=[i]),
            }
        )
    return scoring


def cross_validate(
    clf_lib: str,
    clf,
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    groups: Optional[np.ndarray] = None,
    cv: BaseCrossValidator = None,
    scoring: Union[
        str, list[str], dict[str, ScoreFunction], Callable[..., float]
    ] = "accuracy",
    verbose: int = 0,
    n_jobs: int = 1,
    fit_params: dict[str, Any] = {},
):
    """Cross validate a classifier.

    Parameters
    ----------
    clf_lib: str
        Classifier library to use. Must be one of "sk", "tf", or "pt".
    clf:
        Classifier to cross validate.
    x: np.ndarray
        Input data.
    y: np.ndarray, optional
        Target labels.
    groups: np.ndarray, optional
        Groups to split data into for cross validation.
    cv: BaseCrossValidator, optional
        Cross validation strategy. If None, use StratifiedKFold.
    scoring: str, list, dict, or callable, optional
        Scoring metric(s) to use. If a string, must be a valid
        scikit-learn scorer.
    verbose: int, optional
        Verbosity level.
    n_jobs: int, optional
        Number of jobs to run in parallel. Only used for scikit-learn.
    fit_params: dict, optional
        Parameters to pass to the classifier's fit method.

    Returns
    -------
    result: ExperimentResult
        Cross validation results.
    """
    cross_validate_fn: Callable[..., ExperimentResult]
    if clf_lib == "sk":
        from ertk.sklearn.classification import sk_cross_validate

        # We have to set n_jobs here because using a
        # `with joblib.Parallel...` clause doesn't work properly
        cross_validate_fn = partial(sk_cross_validate, n_jobs=n_jobs)
    elif clf_lib == "tf":
        from ertk.tensorflow.classification import tf_cross_validate

        n_jobs = 1
        cross_validate_fn = tf_cross_validate
    elif clf_lib == "pt":
        from ertk.pytorch.classification import pt_cross_validate

        n_jobs = 1
        cross_validate_fn = pt_cross_validate
    else:
        raise ValueError(f"Unknown classifier type: {clf_lib}")

    result = cross_validate_fn(
        clf,
        x,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        verbose=verbose,
        n_jobs=n_jobs,
        fit_params=fit_params,
    )
    return result


def dataset_cross_validation(
    clf,
    dataset: Dataset,
    clf_lib: str,
    partition: Optional[str] = None,
    label: str = "label",
    cv: Union[BaseCrossValidator, int] = 10,
    verbose: int = 0,
    n_jobs: int = 1,
    scoring: Union[
        str, list[str], dict[str, ScoreFunction], Callable[..., float], None
    ] = None,
    fit_params: dict[str, Any] = {},
) -> ExperimentResult:
    """Cross validates a `Classifier` instance on a single dataset.

    Parameters
    ----------
    clf: class that implements fit() and predict()
        The classifier to test.
    dataset: Dataset
        The dataset for within-corpus cross-validation.
    clf_lib: str
        One of {"sk", "tf", "pt"} to select which library-specific
        cross-validation method to use, since they're not all quite
        compatible.
    partition: str, optional
        The name of the partition to cross-validate over. If None, then
        don't use group cross-validation.
    label: str
        The annotations to use as class labels.
    cv: int or BaseCrossValidator
        A splitter used for cross-validation. Default is KFold(10) for
        10 fold cross-validation.
    verbose: bool
        Passed to cross_validate().
    n_jobs: bool
        Passed to cross_validate().
    scoring: str, list, dict, optional
        Scoring metric(s) to use. Can be anything accepted by
        scikit-learn's cross_val* methods (i.e. str, list or dict).
    fit_params: dict
        Additional parameters passed to the model's fit() method. This
        should be used to pass any more specific parameters not covered
        here.

    Returns
    -------
    df: pandas.DataFrame
        A dataframe holding the results from all runs with this model.
    """
    groups = None if partition is None else dataset.get_group_indices(partition)
    if isinstance(cv, int):
        cv = get_cv_splitter(bool(partition), cv)
    if isinstance(scoring, list) or callable(scoring):
        scoring = make_scoring_dict(scoring)
    if scoring is None or len(scoring) == 0:
        scoring = standard_class_scoring(dataset.get_group_names(label))

    logging.info(f"Starting cross-validation with n_jobs={n_jobs}")
    start_time = time.perf_counter()
    result = cross_validate(
        clf_lib,
        clf,
        dataset.x,
        dataset.get_group_indices(label),
        groups=groups,
        cv=cv,
        scoring=scoring,
        verbose=verbose,
        n_jobs=n_jobs,
        fit_params=fit_params,
    )
    total_time = time.perf_counter() - start_time
    logging.info(f"Cross-validation complete in {total_time:.2f}s")

    index = None
    if isinstance(cv, LeaveOneGroupOut) and partition is not None:
        index = dataset.get_group_names(partition)
    result.scores_df = scores_to_df(result.scores_dict, index=index)
    return result


def dataset_train_val_test(
    clf,
    dataset: Dataset,
    train_idx: Union[Sequence[int], np.ndarray],
    valid_idx: Union[Sequence[int], np.ndarray],
    test_idx: Union[Sequence[int], np.ndarray, None] = None,
    label: str = "label",
    clf_lib: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    verbose: int = 0,
    scoring: Union[
        str, list[str], dict[str, ScoreFunction], Callable[..., float], None
    ] = None,
    fit_params: dict[str, Any] = {},
) -> ExperimentResult:
    """Trains a `Classifier` instance on some training data, optionally
    using validation data, and returns results on given test data.

    Parameters
    ----------
    clf: class that implements fit() and predict()
        The classifier to test.
    dataset: Dataset
        The dataset for within-corpus cross-validation.
    clf_lib: str
        One of {"sk", "tf", "pt"} to select which library-specific
        cross-validation method to use, since they're not all quite
        compatible.
    verbose: bool
        Passed to train_val_test().
    scoring: str, list, dict, optional
        Scoring metric(s) to use. Can be anything accepted by
        scikit-learn's cross_val* methods (i.e. str, list or dict).
    fit_params: dict
        Additional parameters passed to the model's fit() method. This
        should be used to pass any more specific parameters not covered
        here.

    Returns
    -------
    df: pandas.DataFrame
        A dataframe holding the results from all runs with this model.
    """
    if isinstance(scoring, list) or callable(scoring):
        scoring = make_scoring_dict(scoring)
    if scoring is None or len(scoring) == 0:
        scoring = standard_class_scoring(dataset.get_group_names(label))

    train_val_test_fn: Callable[..., ExperimentResult]
    if clf_lib == "sk":
        from ertk.sklearn.classification import sk_train_val_test

        train_val_test_fn = sk_train_val_test
    elif clf_lib == "tf":
        from ertk.tensorflow.classification import tf_train_val_test

        train_val_test_fn = tf_train_val_test
    elif clf_lib == "pt":
        from ertk.pytorch.classification import pt_train_val_test

        train_val_test_fn = pt_train_val_test

    train_idx = np.array(train_idx, copy=False)
    valid_idx = np.array(valid_idx, copy=False)
    if test_idx is None:
        test_idx = valid_idx
    test_idx = np.array(test_idx, copy=False)

    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        raise ValueError("One of {train, val, test} indices are missing.")

    y = dataset.get_group_indices(label)
    train_data: tuple = (dataset.x[train_idx], y[train_idx])
    if sample_weight is not None:
        train_data = train_data + (sample_weight[train_idx],)
    valid_data: tuple = (dataset.x[valid_idx], y[valid_idx])
    if sample_weight is not None:
        valid_data = valid_data + (sample_weight[valid_idx],)
    test_data: tuple = (dataset.x[test_idx], y[test_idx])
    if sample_weight is not None:
        test_data = test_data + (sample_weight[test_idx],)

    start_time = time.perf_counter()
    logging.info("Starting train/val/test.")
    result = train_val_test_fn(
        clf,
        train_data,
        valid_data,
        test_data,
        scoring=scoring,
        verbose=verbose,
        fit_params=fit_params,
    )
    total_time = time.perf_counter() - start_time
    logging.info(f"Train/val/test complete in {total_time:.2f}s")

    result.scores_df = scores_to_df(result.scores_dict)
    return result


def get_balanced_sample_weights(labels: Union[list[int], np.ndarray]):
    """Gets sample weights such that each unique label has the same
    total weight across all instances.

    Parameters
    ----------
    labels: list or array
        Sequence of labels of length n_samples.

    Returns
    -------
    sample_weights: np.ndarray
        Array of sample weights of length n_samples.
    """
    unique, indices, counts = np.unique(labels, return_counts=True, return_inverse=True)
    class_weight = len(labels) / (len(unique) * counts)
    sample_weights = class_weight[indices].astype(np.float32)
    return sample_weights


def get_balanced_class_weights(classes: Union[list[int], np.ndarray]):
    """Gets class weights such that each class has the same total weight
    across all instances.

    Parameters
    ----------
    classes: list or array
        Sequence of classes of length n_samples.

    Returns
    -------
    class_weights: np.ndarray
        Array of class weights of length n_classes.
    """
    unique, counts = np.unique(classes, return_counts=True)
    class_weights = len(classes) / (len(unique) * counts)
    return class_weights


def class_ratings_to_probs(
    ratings: pd.Series, classes: Optional[list[str]] = None
) -> np.ndarray:
    """Convert annotator ratings into distribution over classes for each
    instance.

    Parameters
    ----------
    ratings: pd.Series
        The Pandas `Series` containing the ratings.
    classes: list of str, optional
        Limit to these classes.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (n_instances, n_classes) containing
        probabilities (relative frequencies) for each class.
    """
    ratings = ratings.astype("category")
    if classes:
        ratings = ratings.cat.set_categories(classes)

    mat = ratings.groupby(level=0).value_counts(sort=False).unstack().to_numpy()
    mat = mat / np.sum(mat, axis=1, keepdims=True)
    return mat
