import logging
import time
from typing import Any, Callable, Dict, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
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
from tensorflow.keras.models import Model

from .dataset import LabelledDataset
from .sklearn.classification import sk_cross_validate
from .tensorflow.classification import tf_cross_validate
from .utils import get_cv_splitter


def binary_accuracy_score(
    y_true, y_pred, *, labels=None, average="binary", sample_weight=None
):
    """Calculated binary accuracy. Binary accuracy is the same as
    accuracy considering only a single class.

    Args:
    -----
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

    Returns:
    --------
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
                c
                + "_rec": make_scorer(
                    recall_score, average=None, labels=[i], zero_division=0
                ),
                c
                + "_prec": make_scorer(
                    precision_score, average=None, labels=[i], zero_division=0
                ),
                c
                + "_f1": make_scorer(
                    f1_score, average=None, labels=[i], zero_division=0
                ),
                c + "_ba": make_scorer(binary_accuracy_score, average=None, labels=[i]),
            }
        )
    return scoring


def within_corpus_cross_validation(
    clf: Union[BaseEstimator, Callable[..., Model]],
    dataset: LabelledDataset,
    clf_lib: Optional[str] = None,
    partition: Optional[str] = None,
    cv: Union[BaseCrossValidator, int] = 10,
    verbose: int = 0,
    n_jobs: int = 1,
    fit_params: Dict[str, Any] = {},
) -> pd.DataFrame:
    """Cross validates a `Classifier` instance on a single dataset.

    Parameters:
    -----------
    clf: class that implements fit() and predict()
        The classifier to test.
    dataset: LabelledDataset
        The dataset for within-corpus cross-validation.
    clf_lib: str
        One of {"sk", "tf", "pt"} to select which library-specific
        cross-validation method to use, since they're not all quite
        compatible.
    partition: str, optional
        The name of the partition to cross-validate over. If None, then
        don't use group cross-validation.
    cv: int or BaseCrossValidator
        A splitter used for cross-validation. Default is KFold(10) for
        10 fold cross-validation.
    verbose: bool
        Passed to cross_validate().
    n_jobs: bool
        Passed to cross_validate().

    Returns:
    --------
    df: pandas.DataFrame
        A dataframe holding the results from all runs with this model.
    """
    groups = None if partition is None else dataset.get_group_indices(partition)
    if isinstance(cv, int):
        cv = get_cv_splitter(bool(partition), cv)
    scoring = standard_class_scoring(dataset.classes)

    if clf_lib == "sk":
        cross_validate_fn = sk_cross_validate
    elif clf_lib == "tf":
        n_jobs = 1
        cross_validate_fn = tf_cross_validate

    start_time = time.perf_counter()
    with joblib.Parallel(n_jobs=n_jobs, verbose=verbose):
        scores = cross_validate_fn(
            clf,
            dataset.x,
            dataset.y,
            cv=cv,
            scoring=scoring,
            groups=groups,
            verbose=verbose,
            fit_params=fit_params,
        )
    total_time = time.perf_counter() - start_time
    logging.info(f"Cross-validation complete in {total_time:.2f}s")

    n_folds = len(next(iter(scores.values())))
    if isinstance(cv, LeaveOneGroupOut) and partition is not None:
        index = pd.Index(dataset.get_group_names(partition), name="fold")
    else:
        index = pd.RangeIndex(1, n_folds + 1, name="fold")
    score_df = pd.DataFrame(
        {k[5:]: v for k, v in scores.items() if k.startswith("test_")}, index=index
    )
    return score_df
