import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from ertk.sklearn.utils import get_base_estimator
from ertk.train import get_pipeline_params, get_scores
from ertk.utils import ScoreFunction, filter_kwargs

_logger = logging.getLogger("ertk.sklearn.classification")


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
    _logger.debug(f"cross_validate(): filtered fit_params={fit_params}")

    scores = cross_validate(
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
        return_estimator=True,
    )
    params = []
    for est in scores["estimator"]:
        if isinstance(est, BaseSearchCV):
            params.append(
                json.dumps(get_base_estimator(est.best_estimator_).get_params())
            )
        else:
            params.append(json.dumps(get_base_estimator(est).get_params()))
    scores["params"] = params
    del scores["estimator"]
    return scores


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
    if isinstance(clf, Pipeline):
        fit_params = get_pipeline_params(fit_params, clf)
    else:
        fit_params = filter_kwargs(fit_params, clf.fit)
    _logger.debug(f"train_val_test(): filtered fit_params={fit_params}")

    if test_data is None:
        test_data = valid_data

    if verbose > 0:
        print(f"Fitting classifier {clf}")
    start_time = time.perf_counter()
    clf = clf.fit(train_data[0], train_data[1], **fit_params)
    fit_time = time.perf_counter() - start_time
    if verbose > 0:
        print(f"Getting predictions from classifier {clf}")
    start_time = time.perf_counter()
    y_pred = clf.predict(test_data[0])
    score_time = time.perf_counter() - start_time
    scores: Dict[str, Any] = {
        "fit_time": fit_time,
        "score_time": score_time,
        **get_scores(scoring, y_pred, test_data[1]),
    }
    scores = {f"test_{k}": v for k, v in scores.items()}
    if isinstance(clf, BaseSearchCV):
        scores["params"] = json.dumps(
            get_base_estimator(clf.best_estimator_).get_params()
        )
    else:
        scores["params"] = json.dumps(get_base_estimator(clf).get_params())
    return scores
