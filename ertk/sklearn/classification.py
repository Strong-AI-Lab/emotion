from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline

from ..utils import ScoreFunction


def _filter_kwargs(kwargs: Dict[str, Any], method: Callable) -> Dict[str, Any]:
    """Removes incompatible keyword arguments.

    Args:
    -----
    params: dict
        Keyword arguments to pass to method.
    method: callable
        The method for which to check valid parameters.

    Returns:
    --------
    params: dict
        Filtered keyword arguments.
    """
    import inspect

    meth_params = inspect.signature(method).parameters
    kwargs = kwargs.copy()
    for key in set(kwargs.keys()):
        # Can't use kwargs detection because scikit doesn't support kwargs in general
        if (
            key not in meth_params
            or meth_params[key].kind == inspect.Parameter.POSITIONAL_ONLY
        ):
            del kwargs[key]
    return kwargs


def _get_pipeline_params(params: Dict[str, Any], pipeline: Pipeline) -> Dict[str, Any]:
    """Modifies parameter names to pass to a Pipeline instance."""
    new_params = {}
    for name, est in pipeline.named_steps.items():
        if est is None or est == "passthrough":
            continue
        filt_params = _filter_kwargs(params, est.fit)
        new_params.update({name + "__" + k: v for k, v in filt_params.items()})
    return new_params


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
    fit_params: Dict[str, Any] = {},
):
    if isinstance(clf, Pipeline):
        fit_params = _get_pipeline_params(fit_params, clf)
    else:
        fit_params = _filter_kwargs(fit_params, clf.fit)

    return cross_validate(
        clf,
        x,
        y,
        cv=cv,
        scoring=scoring,
        groups=groups,
        n_jobs=None,
        verbose=verbose,
        fit_params=fit_params,
        error_score="raise",
    )
