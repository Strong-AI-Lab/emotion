"""
Training and evaluation classes and functions
=============================================

Config classes
--------------
.. autosummary::
    :toctree: generated/

    TrainConfig
    ExperimentResult
    ExperimentConfig
    ModelConfig
    CrossValidationConfig
    EvalConfig


Splitting classes and functions
-------------------------------
.. autosummary::
    :toctree: generated/

    TrainValidation
    ShuffleGroupKFold
    ValidationSplit
    get_cv_splitter


Miscellaneous functions
-----------------------
.. autosummary::
    :toctree: generated/

    get_pipeline_params
    get_scores
    scores_to_df
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import numpy as np
import omegaconf
import pandas as pd
from sklearn.base import MetaEstimatorMixin
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

from ertk.config import ERTKConfig
from ertk.dataset.dataset import DataLoadConfig, DataSelector
from ertk.utils import PathOrStr, ScoreFunction, filter_kwargs

__all__ = [
    "TrainConfig",
    "ExperimentResult",
    "ExperimentConfig",
    "ModelConfig",
    "CrossValidationConfig",
    "EvalConfig",
    "TrainValidation",
    "ShuffleGroupKFold",
    "ValidationSplit",
    "get_cv_splitter",
    "get_pipeline_params",
    "get_scores",
    "scores_to_df",
]


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
    """Validation method that uses a pre-defined validation set.

    Parameters
    ----------
    train_idx: list of int or np.ndarray
        Indices of the training set.
    valid_idx: list of int or np.ndarray
        Indices of the validation set.
    """

    def __init__(
        self,
        train_idx: Union[list[int], np.ndarray],
        valid_idx: Union[list[int], np.ndarray],
    ):
        self.train_idx = train_idx
        self.valid_idx = valid_idx

    def split(self, X, y, groups):
        if self.train_idx is not None:
            train_idx = np.array(self.train_idx)
        else:
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
    random_state: Optional[int] = None,
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


def get_pipeline_params(params: dict[str, Any], pipeline: Pipeline):
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
        filt_params = params
        if not isinstance(est, MetaEstimatorMixin):  # May pass through kwargs
            filt_params = filter_kwargs(filt_params, est.fit)
        new_params.update({name + "__" + k: v for k, v in filt_params.items()})
    return new_params


def get_scores(
    scoring: Union[str, list[str], dict[str, ScoreFunction], Callable[..., float]],
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
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
        scoring = {scoring: get_scorer(scoring)}
    elif callable(scoring):
        scoring = {"score": scoring}
    elif isinstance(scoring, list):
        scoring = {x: get_scorer(x) for x in scoring}
    return _score(dummy, None, y_true, scoring)


def scores_to_df(
    scores: dict[str, np.ndarray], index: Union[pd.Index, Iterable, None] = None
):
    """Convert scikit-learn scores dictionary to pandas dataframe.

    Parameters
    ----------
    scores: dict
        Scores dictionary. Each key is a string and all values are
        either scalar or arrays.
    index: pd.Index or iterable, optional
        Optional index to use. Must have the same length as each value
        in `scores`. If not given a `pd.RangeIndex` is used.

    Returns
    -------
    pd.DataFrame
        Scores dataframe, with a column per key in `scores` and row per
        item in each value array.
    """
    _val = next(iter(scores.values()))
    if index is None:
        if np.ndim(_val) == 0:
            index = pd.RangeIndex(1, name="fold")
        else:
            index = pd.RangeIndex(len(_val), name="fold")
    elif not isinstance(index, pd.Index):
        index = pd.Index(index, name="fold")
    return pd.DataFrame(
        {k[5:] if k.startswith("test_") else k: v for k, v in scores.items()},
        index=index,
    )


@dataclass
class ExperimentResult:
    """Class to hold results of an experiment."""

    scores_dict: dict[str, np.ndarray] = field(default_factory=dict)
    """Dictionary of scores. Each key is a string and all values are
    either scalar or arrays.
    """
    scores_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Scores dataframe, with a column per key in `scores_dict` and row
    per evaluation.
    """
    predictions: Optional[np.ndarray] = None
    """Array of predictions. If not given, this is None."""


@dataclass
class ModelConfig(ERTKConfig):
    """Class to hold model configuration."""

    type: str = omegaconf.MISSING
    """Model type. This is used to determine which model class to
    instantiate.
    """
    config: Any = None
    """Model configuration. This is passed to the model class
    constructor.
    """
    args: dict[str, Any] = field(default_factory=dict)
    """Model arguments. These are passed to the model class constructor
    as keyword arguments.
    """
    args_path: Optional[str] = None
    """Path to a YAML file containing model arguments. These are passed
    to the model class constructor as keyword arguments.
    """
    param_grid: dict[str, Any] = field(default_factory=dict)
    """Parameter grid for hyperparameter search. This is passed to
    scikit-learn's `GridSearchCV` class.
    """
    param_grid_path: Optional[str] = None
    """Path to a YAML file containing a parameter grid for
    hyperparameter search. This is passed to scikit-learn's
    `GridSearchCV` class.
    """


@dataclass
class CrossValidationConfig(ERTKConfig):
    """Class to hold cross-validation configuration."""

    part: Optional[str] = None
    """Partition to use for cross-validation. If not given, then
    cross-validation is performed over instances instead of groups.
    """
    kfold: int = omegaconf.MISSING
    """Number of folds for cross-validation."""
    test_size: float = 0.2
    """Fraction of data to use for testing. This is only used if `kfold`
    is 1.
    """


@dataclass
class EvalConfig(ERTKConfig):
    """Class to hold evaluation configuration."""

    cv: Optional[CrossValidationConfig] = None
    """Cross-validation configuration. If not given, then no
    cross-validation is performed.
    """
    train: Optional[DataSelector] = None
    """Data selector for training data."""
    valid: Optional[DataSelector] = None
    """Data selector for validation data."""
    test: Optional[DataSelector] = None
    """Data selector for testing data."""
    inner_kfold: Optional[int] = None
    """Number of folds for inner cross-validation, if using
    hyperparameter search.
    """
    inner_part: Optional[str] = None
    """Partition to use for inner cross-validation, if using
    hyperparameter search.
    """


class TransformClass(Enum):
    std = "std"
    minmax = "minmax"


@dataclass
class TrainConfig(ERTKConfig):
    """Class to hold training configuration."""

    balanced: bool = True
    """Whether to use class-balanced sample weights."""
    reps: int = 1
    """Number of repetitions."""
    normalise: str = "online"
    """Normalisation method. Can be one of "online", or "none"."""
    transform: TransformClass = TransformClass.std
    """Transform method for normalisation. Can be one of "std",
    "minmax", or "none".
    """
    seq_transform: str = "global"
    """Transform method for sequence normalisation."""
    n_jobs: int = 1
    """Number of jobs to run in parallel."""
    verbose: int = 0
    """Verbosity level."""
    label: str = "label"
    """Name of label column."""
    sklearn: Any = None
    """Scikit-learn configuration."""
    pytorch: Any = None
    """PyTorch configuration."""
    tensorflow: Any = None
    """TensorFlow configuration."""


T = TypeVar("T", bound="ExperimentConfig")


@dataclass
class ExperimentConfig(ERTKConfig):
    """Class to hold experiment configuration."""

    name: str = "default"
    """Experiment name."""
    data: DataLoadConfig = omegaconf.MISSING
    """Data loading configuration."""
    model: ModelConfig = omegaconf.MISSING
    """Model configuration."""
    eval: Optional[EvalConfig] = None
    """Evaluation configuration."""
    evals: dict[str, EvalConfig] = field(default_factory=dict)
    """Dictionary of evaluation configurations."""
    training: TrainConfig = field(default_factory=TrainConfig)
    """Training configuration."""
    results: str = ""
    """Path to output results."""
    metrics: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls: type[T], path: PathOrStr) -> T:
        path = Path(path)
        conf = super().from_file(path)
        # TODO: Make more general (i.e. arbitrary keys referencing paths)
        if isinstance(conf.data, str):
            conf.data = DataLoadConfig.from_file(path.parent / conf.data)
        return conf
