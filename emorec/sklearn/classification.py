from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import ParameterGrid

from ..classification import Classifier, ScoreFunction, optimise_params
from ..utils import shuffle_multiple

SKClassifierFunction = Callable[[], ClassifierMixin]


class SKLearnClassifier(Classifier):
    """Class wrapper for a scikit-learn classifier instance.

    Parameters:
    -----------
    model_fn: callable
        A callable that returns a new proper classifier that can be
        trained.
    param_grid: dict, optional
    """

    def __init__(
        self,
        model_fn: SKClassifierFunction,
        param_grid: Optional[Dict[str, Sequence]],
        cv_score_fn: Optional[ScoreFunction],
    ):
        self.model_fn = model_fn
        self.param_grid = ParameterGrid(param_grid)
        self.cv_score_fn = cv_score_fn

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        fold=None,
    ):
        x_train, y_train = shuffle_multiple(x_train, y_train, numpy_indexing=True)
        x_valid, y_valid = shuffle_multiple(x_valid, y_valid, numpy_indexing=True)

        if self.param_grid:
            self.clf = optimise_params(
                self.param_grid,
                self.model_fn,
                self.cv_score_fn,
                x_train,
                y_train,
                x_valid,
                y_valid,
                max_workers=24,
            )
        else:
            self.clf = self.model_fn()
            self.clf.fit(x_train, y_train)

    def predict(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.clf.predict(x_test), y_test
