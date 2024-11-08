"""Utilities for scikit-learn estimators.

.. autosummary::
    :toctree:

    get_estimator_tree
    get_base_estimator
    OneVsRestClassifier
    GridSearchVal
"""

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier as _SKOvR
from sklearn.multiclass import _ConstantPredictor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

__all__ = [
    "get_estimator_tree",
    "get_base_estimator",
    "OneVsRestClassifier",
    "GridSearchVal",
]


# Workaround for OneVsRest
# https://github.com/scikit-learn/scikit-learn/issues/10882#issuecomment-376912896
# https://stackoverflow.com/a/49535681/10044861
def _fit_binary(estimator, X, y, classes=None, **kwargs):
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn(
                "Label %s is present in all training examples." % str(classes[c])
            )
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(X, y, **kwargs)
    return estimator


class OneVsRestClassifier(_SKOvR):
    def fit(self, X, y, **kwargs):
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_binary)(
                self.estimator,
                X,
                column,
                classes=[
                    "not %s" % self.label_binarizer_.classes_[i],
                    self.label_binarizer_.classes_[i],
                ],
                **kwargs,
            )
            for i, column in enumerate(columns)
        )
        return self


def get_estimator_tree(clf) -> list:
    """Gets the estimator list of nested estimators.

    Parameters
    ----------
    clf: estimator
        An estimator.

    Returns
    -------
    list
        The tree of estimators of `clf`.
    """
    tree = [clf]
    if isinstance(clf, Pipeline):
        for x in clf.steps[:-1]:
            tree += get_estimator_tree(x[1])
        if clf._final_estimator != "passthrough":
            return tree + get_estimator_tree(clf._final_estimator)
    for attr in ["base_estimator", "estimator"]:
        if hasattr(clf, attr):
            return tree + get_estimator_tree(getattr(clf, attr))
    return tree


def get_base_estimator(clf) -> BaseEstimator:
    """Gets the base estimator of a pipeline or meta-estimator, assuming
    there is only one.

    Parameters
    ----------
    clf: estimator
        An estimator.

    Returns
    -------
    estimator
        The base estimator of `clf`.
    """
    if isinstance(clf, Pipeline):
        if clf._final_estimator == "passthrough":
            raise ValueError("Pipeline final estimator is 'passthrough'.")
        return get_base_estimator(clf._final_estimator)
    if isinstance(clf, MetaEstimatorMixin):
        for attr in ["base_estimator", "estimator"]:
            if hasattr(clf, attr):
                return get_base_estimator(getattr(clf, attr))
        raise RuntimeError("Couldn't get base estimator from meta estimator")
    return clf


class GridSearchVal(GridSearchCV):
    """Grid-search but using a single validation set. Simple hack of
    GridSearchCV to avoid retraining on train + val data.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=False,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator,
            param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=False,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def fit(self, X, y=None, *, groups=None, **fit_params):
        from sklearn.utils.validation import _check_fit_params

        self.refit = False
        super().fit(X, y=y, groups=groups, **fit_params)

        # Refit only on real train part of "train" data
        train, _ = next(iter(self.cv.split(X, y, groups)))
        fit_params = _check_fit_params(X, fit_params, indices=train)
        self.best_estimator_ = clone(
            clone(self.estimator).set_params(**self.best_params_)
        )
        self.best_estimator_.fit(X[train], y[train], **fit_params)
        return self.best_estimator_
