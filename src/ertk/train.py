from typing import Any, Callable, Dict, List, Union

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection._validation import _score
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array

from ertk.utils import ScoreFunction, filter_kwargs


class TrainValidation(BaseCrossValidator):
    """Validation method that uses the training set as validation set."""

    def split(self, X, y, groups):
        yield np.arange(len(X)), np.arange(len(X))

    def get_n_splits(self, X, y, groups):
        return 1


class ShuffleGroupKFold(_BaseKFold):
    """Like GroupKFold but with random combinations of groups instead of
    deterministic combinations based on group size. This is most useful
    if you have groups of near equal size, and you want group k-fold CV,
    where k divides n_groups.

    Note: If shuffle=False, this does not behave identical to
    GroupKFold, but rather splits groups in sorted order (as returned by
    `numpy.unique()`).
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        # Pairs of start,end indices of groups each of n folds
        fold_idx = np.linspace(0, n_groups, self.n_splits + 1, dtype=int)

        group_order = np.arange(n_groups)
        if self.shuffle:
            # Shuffle order groups appear in folds
            group_order = np.random.default_rng(self.random_state).permutation(
                group_order
            )

        # Mapping from group index to fold index
        group_to_fold = np.zeros(n_groups)

        for fold, (g1, g2) in enumerate(zip(fold_idx[:-1], fold_idx[1:])):
            group_to_fold[group_order[g1:g2]] = fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]


class ValidationSplit(BaseCrossValidator):
    """Validation method that uses a pre-defined validation set."""

    def __init__(self, valid_idx: Union[List[int], np.ndarray]):
        self.valid_idx = valid_idx

    def split(self, X, y, groups):
        train_idx = np.arange(len(X))
        train_idx = train_idx[~np.isin(train_idx, self.valid_idx)]
        yield train_idx, self.valid_idx

    def get_n_splits(self, X, y, groups):
        return 1


def get_cv_splitter(
    group: bool,
    k: int,
    test_size: float = 0.2,
    shuffle: bool = False,
    random_state: int = None,
) -> BaseCrossValidator:
    """Gets an appropriate cross-validation splitter for the given
    number of folds and groups, or a single random split.

    Parameters
    ----------
    group: bool
        Whether to split over pre-defined groups of instances.
    k: int
        If k > 1 then do k-fold CV. If k == 1 then do one random
        split. If k = -1 then do leave-one-out. If k == 0 then use the
        whole train set as validation split.
    test_size: float
        The size of the test set when k == 1 (one random split).
    shuffle: bool
        Whether to shuffle when using k-fold for k > 1.
    random_state: int, optional
        The random state to set for splitters with shuffling behaviour.

    Returns
    -------
    splitter: BaseCrossValidator
        Cross-validation splitter that has `split()` and
        `get_n_splits()` methods.
    """
    # TODO: Leave-|k|-out for k < 0?
    if k == 0:
        return TrainValidation()
    if group:
        if k > 1:
            if shuffle:
                return ShuffleGroupKFold(k, shuffle=shuffle, random_state=random_state)
            return GroupKFold(k)
        elif k == 1:
            return GroupShuffleSplit(1, test_size=test_size, random_state=random_state)
        return LeaveOneGroupOut()
    if k > 1:
        return StratifiedKFold(k, shuffle=shuffle, random_state=random_state)
    elif k == 1:
        return StratifiedShuffleSplit(1, test_size=test_size, random_state=random_state)
    return LeaveOneOut()


def get_pipeline_params(params: Dict[str, Any], pipeline: Pipeline):
    """Modifies parameter names to pass to a Pipeline instance's `fit()`
    method.

    Parameters
    ----------
    params: dict
        Parameters to pass to Pipeline.fit(). All parameters are passed
        to all estimators in the pipeline so long as they are valid.
    pipeline: Pipeline
        The pipeline instance.

    Returns
    -------
    new_params: dict
        Parameters filtered and prepended with pipeline step names and
        double underscore (e.g. groups -> clf__groups).
    """
    new_params = {}
    for name, est in pipeline.named_steps.items():
        if est is None or est == "passthrough":
            continue
        filt_params = filter_kwargs(params, est.fit)
        new_params.update({name + "__" + k: v for k, v in filt_params.items()})
    return new_params


def get_scores(
    scoring: Union[str, List[str], Dict[str, ScoreFunction], Callable[..., float]],
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    """Get dictionary of scores for predictions.

    Parameters
    ----------
    scoring: str, list, dict or callable
        Score(s) to calculate. This takes the same for as for
        scikit-learn's cross_val_* methods.
    y_pred: array-like
        Predictions.
    y_true: array-like
        Ground truth.

    Returns
    -------
    scores: dict
        A dictionary mapping score names to corresponding score(s).
    """

    class DummyEstimator:
        """Class that implements a dummy estimator for scoring, to avoid
        repeated invocations of `predict()` etc.
        """

        def __init__(self, y_pred):
            self.y_pred = y_pred

        def predict(self, x, **kwargs):
            return self.y_pred

        def predict_proba(self, x, **kwargs):
            return self.y_pred

        def decision_function(self, x, **kwargs):
            return self.y_pred

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    dummy = DummyEstimator(y_pred)

    if isinstance(scoring, str):
        scoring = {"score": get_scorer(scoring)}
    elif callable(scoring):
        scoring = {"score": scoring}
    elif isinstance(scoring, list):
        scoring = {x: get_scorer(x) for x in scoring}
    return _score(dummy, None, y_true, scoring)
