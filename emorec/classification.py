from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    LeaveOneGroupOut,
    cross_validate,
)
from sklearn.utils.multiclass import unique_labels

from .dataset import LabelledDataset


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


def _filter_params(params: Dict[str, Any], obj: object, method: str) -> Dict[str, Any]:
    import inspect

    sig = inspect.signature(getattr(obj, method))
    params = params.copy()
    for key in set(params.keys()):
        # Can't use kwargs detection because scikit doesn't support kwargs in general
        if (
            key not in sig.parameters
            or sig.parameters[key].kind == inspect.Parameter.POSITIONAL_ONLY
        ):
            del params[key]
    return params


def _get_pipeline_params(params: Dict[str, Any], pipeline: Pipeline) -> Dict[str, Any]:
    new_params = {}
    for name, est in pipeline.named_steps.items():
        if est is None or est == "passthrough":
            continue
        filt_params = _filter_params(params, est, "fit")
        new_params.update({name + "__" + k: v for k, v in filt_params.items()})
    return new_params


def within_corpus_cross_validation(
    clf: ClassifierMixin,
    dataset: LabelledDataset,
    partition: Optional[str] = None,
    sample_weights: np.ndarray = None,
    reps: int = 1,
    splitter: BaseCrossValidator = KFold(10),
    verbose: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Cross validates a `Classifier` instance on a single dataset.

    Parameters:
    -----------
    clf: class that implements fit() and predict()
        The classifier to test.
    dataset: LabelledDataset
        The dataset for within-corpus cross-validation.
    partition: str, optional
        The name of the partition to cross-validate over. If None, then
        don't use group cross-validation.
    reps: int
        The number of repetitions, default is 1 for a single run.
    splitter: sklearn.model_selection.BaseCrossValidator
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
    group_indices = None if partition is None else dataset.get_group_indices(partition)
    x = dataset.x

    scoring = standard_class_scoring(dataset.classes)
    score_dfs = []
    for rep in range(1, reps + 1):
        print(f"Rep {rep}/{reps}")

        fit_params = {"sample_weight": sample_weights, "groups": group_indices}
        if isinstance(clf, Pipeline):
            fit_params = _get_pipeline_params(fit_params, clf)
        else:
            fit_params = _filter_params(fit_params, clf, "fit")

        scores = cross_validate(
            clf,
            x,
            dataset.y,
            cv=splitter,
            scoring=scoring,
            groups=group_indices,
            n_jobs=n_jobs,
            verbose=verbose,
            fit_params=fit_params,
        )

        n_folds = len(next(iter(scores.values())))
        if isinstance(splitter, LeaveOneGroupOut) and partition is not None:
            index = pd.Index(dataset.get_group_names(partition), name="fold")
        else:
            index = pd.RangeIndex(1, n_folds + 1, name="fold")
        score_df = pd.DataFrame(
            {k[5:]: v for k, v in scores.items() if k.startswith("test_")},
            index=index,
        )
        score_dfs.append(score_df)

    return pd.concat(score_dfs, keys=list(range(1, reps + 1)), names=["rep"])
