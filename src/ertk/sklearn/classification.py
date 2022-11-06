import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from ertk.sklearn.utils import get_base_estimator, get_estimator_tree
from ertk.train import ExperimentResult, get_pipeline_params, get_scores
from ertk.utils import ScoreFunction

logger = logging.getLogger("ertk.sklearn.classification")


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
) -> ExperimentResult:
    logger.debug(f"cross_validate(): filtered fit_params={fit_params}")
    logger.info(f"cross_validate(): verbose={verbose}")

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
    scores["params"] = np.array(params)
    del scores["estimator"]
    return ExperimentResult(scores)


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
) -> ExperimentResult:
    sw_kw = {"sample_weight": train_data[2] if len(train_data) > 2 else None}
    for est in get_estimator_tree(clf):
        if isinstance(est, Pipeline):
            sw_kw = get_pipeline_params(sw_kw, est)
            break
    fit_params = dict(**fit_params, **sw_kw)
    logger.debug(f"train_val_test(): filtered fit_params={fit_params}")

    logger.info(f"Fitting classifier {clf}")
    start_time = time.perf_counter()
    clf = clf.fit(train_data[0], train_data[1], **fit_params)
    fit_time = time.perf_counter() - start_time
    logger.info(f"Getting predictions from classifier {clf}")
    start_time = time.perf_counter()
    y_pred = clf.predict(test_data[0])
    score_time = time.perf_counter() - start_time
    scores: Dict[str, Any] = {
        "fit_time": fit_time,
        "score_time": score_time,
        **get_scores(scoring, y_pred, test_data[1]),
    }
    scores = {f"test_{k}": v for k, v in scores.items()}
    print(confusion_matrix(test_data[1], y_pred))
    if isinstance(clf, BaseSearchCV):
        scores["params"] = json.dumps(
            get_base_estimator(clf.best_estimator_).get_params()
        )
    else:
        scores["params"] = json.dumps(get_base_estimator(clf).get_params())
    return ExperimentResult(scores, predictions=y_pred)
