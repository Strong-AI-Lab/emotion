"""Scikit-learn implementations of various models"""

from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .svm import PrecomputedSVC

CLASSIFIER_MAP = {
    "svm": PrecomputedSVC,
    "rf": RandomForestClassifier,
    "knn": KNeighborsClassifier,
    "lr": LogisticRegression,
}


def get_sk_model_fn(name: str) -> Callable[..., BaseEstimator]:
    """Get a scikit-learn model class.

    Parameters
    ----------
    name: str
        The model name.

    Returns
    -------
    model_fn: callable
        A method that takes arguments and returns a Model instance.
    """
    return CLASSIFIER_MAP[name]


def get_sk_model(name: str, **kwargs) -> BaseEstimator:
    """Get a scikit-learn model by name.

    Parameters
    ----------
    name: str
        The model name.

    Returns
    -------
    model: BaseEstimator
        The unfit scikit-learn classifier.
    """
    return get_sk_model_fn(name)(**kwargs)
