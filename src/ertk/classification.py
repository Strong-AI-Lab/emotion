import logging
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import BaseCrossValidator, LeaveOneGroupOut
from sklearn.utils.multiclass import unique_labels

from ertk.dataset import Dataset
from ertk.train import get_cv_splitter, scores_to_df


def binary_accuracy_score(
    y_true, y_pred, *, labels=None, average="binary", sample_weight=None
):
    """Calculated binary accuracy. Binary accuracy is the same as
    accuracy considering only a single class.

    Parameters
    ----------
    y_true:
        Ground truth labels.
    y_pred:
        Predicted labels.
    labels: list, optional
        Labels to include for average != "binary". If None, all unique
        labels in y_true or y_pred are included.
    average: str, optional
        Method to compute average. If "binary" then simply return
        accuracy. If "macro" then return mean binary accuracy. If
        "weighted" then weight the mean binary accuracy by ground truth
        support. If None then return an array of values, one for each
        label in labels.

    Returns
    -------
    label_accs: float or list
        Binary accuracies for labels in labels or average if `average`
        is not None.
    """
    all_labels = unique_labels(y_true, y_pred)
    if average == "binary":
        if len(all_labels) != 2:
            raise ValueError("Must only have two labels when `average` is 'binary'.")
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if labels is None:
        labels = all_labels
    accs = {l: 0 for l in labels}
    for lab in all_labels:
        acc = accuracy_score(y_true == lab, y_pred == lab, sample_weight=sample_weight)
        accs[lab] = acc
    label_accs = [accs[l] for l in labels]
    if average == "macro":
        return np.mean(label_accs)
    elif average == "weighted":
        counts = [np.count_nonzero(y_true == l) for l in labels]
        return np.average(label_accs, weights=counts)
    elif average is None:
        return label_accs[0] if len(label_accs) == 1 else label_accs


def standard_class_scoring(classes: Sequence[str]):
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


def dataset_cross_validation(
    clf,
    dataset: Dataset,
    clf_lib: Optional[str] = None,
    partition: Optional[str] = None,
    label: str = "label",
    cv: Union[BaseCrossValidator, int] = 10,
    verbose: int = 0,
    n_jobs: int = 1,
    scoring=None,
    fit_params: Dict[str, Any] = {},
) -> pd.DataFrame:
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
    if scoring is None:
        scoring = standard_class_scoring(dataset.get_group_names(label))

    cross_validate_fn: Callable[..., Dict[str, Any]]
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

    start_time = time.perf_counter()
    logging.info(f"Starting cross-validation with n_jobs={n_jobs}")
    scores = cross_validate_fn(
        clf,
        dataset.x,
        dataset.get_group_indices(label),
        cv=cv,
        scoring=scoring,
        groups=groups,
        verbose=verbose,
        fit_params=fit_params,
    )
    total_time = time.perf_counter() - start_time
    logging.info(f"Cross-validation complete in {total_time:.2f}s")

    index = None
    if isinstance(cv, LeaveOneGroupOut) and partition is not None:
        index = dataset.get_group_names(partition)
    return scores_to_df(scores, index=index)


def train_val_test(
    clf,
    dataset: Dataset,
    train_idx: Union[Sequence[int], np.ndarray],
    valid_idx: Union[Sequence[int], np.ndarray],
    test_idx: Union[Sequence[int], np.ndarray, None] = None,
    label: str = "label",
    clf_lib: Optional[str] = None,
    sample_weight: np.ndarray = None,
    verbose: int = 0,
    scoring=None,
    fit_params: Dict[str, Any] = {},
) -> pd.DataFrame:
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
    if scoring is None:
        scoring = standard_class_scoring(dataset.get_group_names(label))

    train_val_test_fn: Callable[..., Dict[str, Any]]
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
    train_data: Tuple = (dataset.x[train_idx], y[train_idx])
    if sample_weight is not None:
        train_data = train_data + (sample_weight[train_idx],)
    valid_data: Tuple = (dataset.x[valid_idx], y[valid_idx])
    if sample_weight is not None:
        valid_data = valid_data + (sample_weight[valid_idx],)
    test_data: Tuple = (dataset.x[test_idx], y[test_idx])
    if sample_weight is not None:
        test_data = test_data + (sample_weight[test_idx],)

    start_time = time.perf_counter()
    logging.info("Starting train/val/test.")
    scores = train_val_test_fn(
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

    return scores_to_df(scores)


def get_balanced_sample_weights(labels: Union[List[int], np.ndarray]):
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
