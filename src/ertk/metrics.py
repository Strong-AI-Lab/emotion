"""Metrics and scores


.. autosummary::
    :toctree: generated/

    binary_accuracy_score
"""


import copy
from collections.abc import Callable
from typing import Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, get_scorer, make_scorer
from sklearn.utils.multiclass import unique_labels

__all__ = [
    "binary_accuracy_score",
]


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


_SCORERS = {
    "uar": get_scorer("balanced_accuracy"),
    "war": get_scorer("accuracy"),
    "microf1": make_scorer(f1_score, average="micro"),
    "macrof1": make_scorer(f1_score, average="macro"),
    "binary_accuracy": make_scorer(binary_accuracy_score),
}


def get_metric(metric: str) -> Callable:
    """Gets the metric callable associated with `metric`.

    Parameters
    ----------
    metric: str
        The metric's name.

    Returns
    -------
    callable
        The function that scores a given estimator and data.
    """
    try:
        return get_scorer(metric)
    except ValueError:
        return copy.deepcopy(_SCORERS[metric])


def make_scoring_dict(scoring: Union[list[str], Callable]) -> dict[str, Callable]:
    """Make a list of scorer names into the equivalent dict.

    Parameters
    ----------
    scoring: list[str] or callable
        list of scorers by name, or single scoring function.

    Returns
    -------
    dict[str, callable]
        The equivalent scoring dictionary.
    """
    if callable(scoring):
        return {str(scoring): scoring}
    return {str(k): get_metric(k) for k in scoring}
