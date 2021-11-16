from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline

from ..utils import ScoreFunction, filter_kwargs, get_pipeline_params, get_scores


def sk_cross_validate(
    clf: ClassifierMixin,
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    groups: Optional[np.ndarray] = None,
    cv: BaseCrossValidator = None,
    scoring: Union[
        str, List[str], Dict[str, ScoreFunction], Callable[..., float]
    ] = "accuracy",
    verbose: int = 0,
    n_jobs: int = 1,
    fit_params: Dict[str, Any] = {},
):
    if isinstance(clf, Pipeline):
        fit_params = get_pipeline_params(fit_params, clf)
    else:
        fit_params = filter_kwargs(fit_params, clf.fit)

    return cross_validate(
        clf,
        x,
        y,
        cv=cv,
        scoring=scoring,
        groups=groups,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_params=fit_params,
        error_score="raise",
    )


def sk_train_val_test(
    clf: ClassifierMixin,
    train_data: Tuple[np.ndarray, ...],
    valid_data: Tuple[np.ndarray, ...],
    test_data: Optional[Tuple[np.ndarray, ...]] = None,
    verbose: int = 0,
    scoring: Union[
        str, List[str], Dict[str, ScoreFunction], Callable[..., float]
    ] = "accuracy",
    fit_params: Dict[str, Any] = {},
):
    fit_params["verbose"] = verbose
    if isinstance(clf, Pipeline):
        fit_params = get_pipeline_params(fit_params, clf)
    else:
        fit_params = filter_kwargs(fit_params, clf.fit)

    if test_data is None:
        test_data = valid_data

    clf = clf.fit(train_data[0], train_data[1], **fit_params)
    y_pred = clf.predict(test_data[0])
    scores = get_scores(scoring, y_pred, test_data[1])
    scores = {f"test_{k}": v for k, v in scores.items()}
    return scores
