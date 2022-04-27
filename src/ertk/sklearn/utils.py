import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.multiclass import OneVsRestClassifier as _SKOvR
from sklearn.multiclass import _ConstantPredictor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer


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


def get_base_estimator(clf):
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
