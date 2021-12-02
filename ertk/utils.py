"""Various utility classes and functions."""

from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import click
import joblib
import numpy as np
import tqdm
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
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
from sklearn.utils.validation import check_array

PathOrStr = Union[PathLike, str]


# Class adapted from user394430's answer here:
# https://stackoverflow.com/a/61900501/10044861
# Licensed under CC BY-SA 4.0
class TqdmParallel(joblib.Parallel):
    """Convenience class that acts identically to joblib.Parallel except
    it uses a tqdm progress bar.
    """

    def __init__(
        self,
        total: int = 1,
        desc: str = "",
        unit: str = "it",
        leave: bool = True,
        **kwargs,
    ):
        self.total = total
        self.tqdm_args = {"desc": desc, "unit": unit, "leave": leave, "disable": None}
        kwargs["verbose"] = 0
        super().__init__(**kwargs)

    def __call__(self, iterable):
        with tqdm.tqdm(total=self.total, **self.tqdm_args) as self.pbar:
            return super().__call__(iterable)

    def print_progress(self):
        self.pbar.n = self.n_completed_tasks
        self.pbar.refresh()


class PathlibPath(click.Path):
    """Convenience class that acts identically to `click.Path` except it
    converts the value to a `pathlib.Path` object.
    """

    def convert(self, value, param, ctx) -> Path:
        return Path(super().convert(value, param, ctx))


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def itmap(s: Callable[[T1], T2]):
    """Returns a new map function that additionally maps tuples to
    tuples and lists to lists.
    """

    @overload
    def _map(x: T1) -> T2:
        ...

    @overload
    def _map(x: List[T1]) -> List[T2]:
        ...

    @overload
    def _map(x: Tuple[T1, ...]) -> Tuple[T2, ...]:
        ...

    def _map(x):
        if isinstance(x, (list, tuple)):
            return type(x)(s(y) for y in x)
        else:
            return s(x)

    return _map


def ordered_intersect(a: Iterable, b: Container) -> List:
    """Returns a list of the intersection of `a` and `b`, in the order
    elements appear in `a`.
    """
    return [x for x in a if x in b]


def filter_kwargs(kwargs: Dict[str, Any], method: Callable) -> Dict[str, Any]:
    """Removes incompatible keyword arguments. This ignores any **kwargs
    catchall in method signature, and only returns args specifically
    present as keyhwords in the method signature which are also not
    positional only.

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
        if (
            key not in meth_params
            or meth_params[key].kind == inspect.Parameter.POSITIONAL_ONLY
        ):
            del kwargs[key]
    return kwargs


def get_arg_mapping_multi(s: str) -> Dict[str, List[Any]]:
    """Given a string mapping from the command-line, returns a dict
    representing that mapping.

    The string form of the mapping is:
        key:value[,key:value]+
    Duplicate keys will be mapped to a list of values.

    Args:
    -----
    s: str
        String representing the mapping. It cannot contain spaces or
        shell symbols (unless escaped).

    Returns:
    --------
    mapping: dict
        A dictionary mapping keys to lists of values from the string.
    """
    mapping: Dict[str, List[str]] = {}
    for cls in s.split(","):
        key, val = cls.split(":")
        if key in mapping:
            mapping[key].append(val)
        else:
            mapping[key] = [val]
    return mapping


def get_arg_mapping(s: Union[Path, str]) -> Dict[str, Any]:
    """Given a mapping on the command-line, returns a dict representing
    that mapping. Mapping can be a string or a more complex YAML file.

    The string form of the mapping is:
        key:value[,key:value]+

    Args:
    -----
    s: PathLike or str
        String representing the mapping or path to YAML containing
        mapping. If a string, it cannot contain spaces or shell symbols
        (unless escaped).

    Returns:
    --------
    mapping: dict
        A dictionary mapping keys to values from the string.
    """
    if isinstance(s, Path) or Path(s).exists():
        with open(s) as fid:
            return yaml.safe_load(fid) or {}
    return {k: v[0] if len(v) == 1 else v for k, v in get_arg_mapping_multi(s).items()}


def flat_to_inst(x: np.ndarray, slices: Union[np.ndarray, List[int]]) -> np.ndarray:
    """Takes a concatenated 2D data array and converts it to either a
    contiguous 2D/3D array or a variable-length 3D array, with one
    feature vector/matrix per instance.
    """

    if len(x) == len(slices):
        # 2-D contiguous array
        return x
    elif all(x == slices[0] for x in slices):
        # 3-D contiguous array
        assert len(x) % len(slices) == 0
        return x.reshape(len(slices), len(x) // len(slices), x[0].shape[-1])
    else:
        # 3-D variable length array
        start_idx = np.cumsum(slices)[:-1]
        return np.array(np.split(x, start_idx, axis=0), dtype=object)


def inst_to_flat(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The inverse of flat_to_inst(). Takes an instance matrix and
    converts to a "flattened" 2D matrix.
    """

    slices = np.ones(len(x), dtype=int)
    if len(x.shape) != 2:
        slices = np.array([len(_x) for _x in x])
        if len(x.shape) == 3:
            x = x.reshape(sum(slices), x.shape[2])
        else:
            x = np.concatenate(x)
    assert sum(slices) == len(x)
    return x, slices


def check_3d(arrays: Union[Sequence[np.ndarray], np.ndarray]):
    """Checks if an array is 3D or each array in a list is 2D. Raises an
    exception if this isn't the case.
    """
    if any(len(x.shape) != 2 for x in arrays):
        raise ValueError("arrays must be 3D (contiguous or vlen).")


def frame_arrays(
    arrays: Union[List[np.ndarray], np.ndarray],
    frame_size: int = 640,
    frame_shift: int = 160,
    num_frames: Optional[int] = None,
):
    """Creates sequences of frames from the given arrays. Each input
    array is a 1-D or L x 1 time domain signal. Each corresponding
    output array is a 2-D array of frames of shape (num_frames,
    frame_size).
    """
    # TODO: Make option for vlen output
    if num_frames is None:
        max_len = max(len(x) for x in arrays)
        num_frames = (max_len - frame_size) // frame_shift + 1

    _arrs = []
    for seq in arrays:
        seq = np.squeeze(seq)
        arr = np.zeros((num_frames, frame_size), dtype=np.float32)
        for i in range(0, len(seq), frame_shift):
            idx = i // frame_shift
            if idx >= num_frames:
                break
            maxl = min(len(seq) - i, frame_size)
            arr[idx, :maxl] = seq[i : i + frame_size]
        _arrs.append(arr)
    arrs = np.array(_arrs)
    assert tuple(arrs.shape) == (len(arrays), num_frames, frame_size)
    return arrs


def pad_arrays(arrays: Union[List[np.ndarray], np.ndarray], pad: int = 32):
    """Pads each array to the nearest multiple of `pad` greater than the
    array size. Assumes axis 0 of each sub-array, or axis 1 of x, is
    the time axis.
    """
    if isinstance(arrays, np.ndarray) and len(arrays.shape) > 1:
        padding = int(np.ceil(arrays.shape[1] / pad)) * pad - arrays.shape[1]
        extra_dims = tuple((0, 0) for _ in arrays.shape[2:])
        return np.pad(arrays, ((0, 0), (0, padding)) + extra_dims)
    new_arrays = []
    for x in arrays:
        padding = int(np.ceil(x.shape[0] / pad)) * pad - x.shape[0]
        new_arrays.append(np.pad(x, ((0, padding), (0, 0))))
    if isinstance(arrays, np.ndarray):
        if all(x.shape == new_arrays[0].shape for x in new_arrays):
            return np.array(new_arrays)
        return np.array(new_arrays, dtype=object)
    return new_arrays


def clip_arrays(
    arrays: Union[List[np.ndarray], np.ndarray], length: int, copy: bool = True
):
    """Clips each array to the specified maximum length."""
    if isinstance(arrays, np.ndarray):
        if len(arrays.shape) > 1:
            return arrays[:, :length, ...].copy() if copy else arrays[:, :length, ...]
        new_arrays = [x[:length].copy() if copy else x[:length] for x in arrays]
        if all(x.shape == new_arrays[0].shape for x in new_arrays):
            # Return contiguous array
            return np.stack(new_arrays)
        return np.array(new_arrays, dtype=object)
    return [x[:length].copy() if copy else x[:length] for x in arrays]


def transpose_time(arrays: Union[List[np.ndarray], np.ndarray]):
    """Transpose the time and feature axis of each array. Requires each
    array be 2-D.

    NOTE: This function modifies the arrays in-place.
    """
    check_3d(arrays)
    if isinstance(arrays, np.ndarray) and len(arrays.shape) == 3:
        arrays = arrays.transpose(0, 2, 1)
    else:
        for i in range(len(arrays)):
            arrays[i] = arrays[i].transpose()
    assert all(x.shape[0] == arrays[0].shape[0] for x in arrays)
    return arrays


def shuffle_multiple(*arrays: Union[np.ndarray, Sequence], numpy_indexing: bool = True):
    """Shuffles multiple arrays or lists in sync. Useful for shuffling the data
    and labels in a dataset separately while keeping them synchronised.

    Parameters:
    -----------
    arrays, iterable of array-like
        The arrays to shuffle. They must all have the same size of first
        dimension.
    numpy_indexing: bool, default = True
        Whether to use NumPy-style indexing or list comprehension.

    Returns:
    shuffled_arrays: iterable of array-like
        The shuffled arrays.
    """
    if any(len(arrays[0]) != len(x) for x in arrays):
        raise ValueError("Not all arrays have equal first dimension.")

    perm = np.random.default_rng().permutation(len(arrays[0]))
    new_arrays = [
        array[perm] if numpy_indexing else [array[i] for i in perm] for array in arrays
    ]
    return new_arrays


def batch_arrays(
    arrays_x: Union[np.ndarray, List[np.ndarray]],
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    uniform_batch_size: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batches a list of arrays of different sizes, grouping them by
    size. This is designed for use with variable length sequences. Each
    batch will have a maximum of batch_size arrays, but may have less if
    there are fewer arrays of the same length. It is recommended to use
    the pad_arrays() method of the LabelledDataset instance before using
    this function, in order to quantise the lengths.

    Parameters:
    -----
    arrays_x: list of ndarray
        A list of N-D arrays, possibly of different lengths, to batch.
        The assumption is that all the arrays have the same rank and
        only axis 0 differs in length.
    y: ndarray
        The labels for each of the arrays in arrays_x.
    batch_size: int
        Arrays will be grouped together by size, up to a maximum of
        batch_size, after which a new batch will be created. Thus each
        batch produced will have between 1 and batch_size items.
    shuffle: bool, default = True
        Whether to shuffle array order in a batch.
    uniform_batch_size: bool, default = False
        Whether to keep all batches the same size, batch_size, and pad
        with zeros if necessary, or have batches of different sizes if
        there aren't enough sequences to group together.

    Returns:
    --------
    x_list: ndarray,
        The batched arrays. x_list[i] is the i'th
        batch, having between 1 and batch_size items, each of length
        lengths[i].
    y_list: ndarray
        The batched labels corresponding to sequences in x_list.
        y_list[i] has the same length as x_list[i].
    """
    if isinstance(arrays_x, list):
        arrays_x = np.array(arrays_x, dtype=object)
    if shuffle:
        arrays_x, y = shuffle_multiple(arrays_x, y, numpy_indexing=False)

    fixed_shape = arrays_x[0].shape[1:]
    lengths = [x.shape[0] for x in arrays_x]
    unique_len = np.unique(lengths)
    x_dtype = arrays_x[0].dtype
    y_dtype = y.dtype

    xlist = []
    ylist = []
    for length in unique_len:
        idx = np.nonzero(lengths == length)[0]
        for b in range(0, len(idx), batch_size):
            batch_idx = idx[b : b + batch_size]
            size = batch_size if uniform_batch_size else len(batch_idx)
            _x = np.zeros((size, length) + fixed_shape, dtype=x_dtype)
            _y = np.zeros(size, dtype=y_dtype)
            _y[:size] = y[batch_idx]
            for i, j in enumerate(batch_idx):
                _x[i, ...] = arrays_x[j]
            xlist.append(_x)
            ylist.append(_y)
    x_batch = np.array(xlist, dtype=object)
    y_batch = np.array(ylist, dtype=y_dtype if uniform_batch_size else object)
    return x_batch, y_batch


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

    Parameters:
    -----------
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

    Returns:
    --------
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


def group_transform(
    x: np.ndarray,
    groups: np.ndarray,
    transform: TransformerMixin,
    *,
    inplace: bool = False,
    **fit_params,
):
    """Per-group (offline) transformation (e.g. standardisation).

    Args:
    -----
    x: np.ndarray
        The data matrix to transform. Each x[i] must be an instance.
    groups: np.ndarray
        Groups assignment for each instance. It must be the case that
        len(groups) == len(x).
    transform:
        The transformation to apply. Must implement fit_transform().
    inplace: bool
        Whether to modify x in-place. Default is False so that a copy is
        made.
    **fit_params:
        Other keyword arguments to pass to the transform.fit() method.

    Returns:
    --------
    x: np.ndarray
        The modified data matrix with transformations applied to each
        group individually.
    """
    if not inplace:
        x = x.copy()
    unique_groups = np.unique(groups)
    for g_id in unique_groups:
        flat, slices = inst_to_flat(x[groups == g_id])
        flat = transform.fit_transform(flat, y=None, **fit_params)
        if len(x.shape) == 1 and len(slices) == 1:
            # Special case to avoid issues for vlen arrays
            _arr = np.empty(1, dtype=object)
            _arr[0] = flat
            x[groups == g_id] = _arr
            continue
        x[groups == g_id] = flat_to_inst(flat, slices)
    return x


def instance_transform(
    x: np.ndarray, transform: TransformerMixin, *, inplace: bool = False, **fit_params
):
    """Per-instance transformation (e.g. standardisation).

    Args:
    -----
    x: np.ndarray
        The data matrix to transform. Each x[i] must be a 2D instance.
    transform:
        The transformation to apply. Must implement fit_transform().
    inplace: bool
        Whether to modify x in-place. Default is False so that a copy is
        made.
    **fit_params:
        Other keyword arguments to pass to the transform.fit() method.

    Returns:
    --------
    x: np.ndarray
        The modified data matrix with transformations applied to each
        instance individually.
    """
    return group_transform(
        x, np.arange(len(x)), transform, inplace=inplace, **fit_params
    )


ScoreFunction = Callable[[np.ndarray, np.ndarray], float]


def get_scores(
    scoring: Union[str, List[str], Dict[str, ScoreFunction], Callable[..., float]],
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    """Get dictionary of scores for predictions.

    Parameters:
    -----------
    scoring: str, list, dict or callable
        Score(s) to calculate. This takes the same for as for
        scikit-learn's cross_val_* methods.
    y_pred: array-like
        Predictions.
    y_true: array-like
        Ground truth.

    Returns:
    --------
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


def get_pipeline_params(params: Dict[str, Any], pipeline: Pipeline):
    """Modifies parameter names to pass to a Pipeline instance's `fit()`
    method.

    Parameters:
    -----------
    params: dict
        Parameters to pass to Pipeline.fit(). All parameters are passed
        to all estimators in the pipeline so long as they are valid.
    pipeline: Pipeline
        The pipeline instance.

    Returns:
    --------
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


class GroupTransformWrapper(TransformerMixin, BaseEstimator):
    """Transform that modifies groups independently without storing
    parameters.
    """

    def __init__(self, transformer: TransformerMixin) -> None:
        self.transformer = transformer

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, groups=None, **fit_params):
        return group_transform(X, groups, self.transformer, inplace=False, **fit_params)


class InstanceTransformWrapper(TransformerMixin, BaseEstimator):
    """Transform that modifies instances independently without storing
    parameters.
    """

    def __init__(self, transformer: TransformerMixin) -> None:
        self.transformer = transformer

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **fit_params):
        raise instance_transform(X, self.transformer, inplace=False, **fit_params)


class SequenceTransform(TransformerMixin, BaseEstimator):
    """Transform designed to process sequences of vectors."""

    pass


class SequenceTransformWrapper(SequenceTransform):
    """Wrapper around a scikit-learn transform that can process
    sequences of vectors.

    Args:
    -----
    transformer:
        An object which implements the fit_transform() method on a
        collection of 1D vectors.
    method: str
        The method to manipuate the sequence into 1D vectors, one of
        {"feature", "global"}. If "feature" then each feature column of
        the concatenated (2D) input is transformed independently. If
        "global" then the transformer is fitted over the whole input
        including all feature columns.
    """

    def __init__(self, transformer: TransformerMixin, method: str):
        VALID_METHODS = {"feature", "global"}
        self.transformer = transformer
        if method not in VALID_METHODS:
            raise ValueError(f"method '{method}' not in {VALID_METHODS}.")
        self.method = method

    def fit(self, X, y=None, **fit_params):
        flat_x, _ = inst_to_flat(X)
        if self.method == "feature":
            self.transformer.fit(flat_x, y=y, **fit_params)
        elif self.method == "global":
            self.transformer.fit(flat_x.reshape((-1, 1)), y=y, **fit_params)
        return self

    def transform(self, X, **fit_params):
        flat_x, slices = inst_to_flat(X)
        if self.method == "feature":
            flat_x = self.transformer.transform(flat_x, **fit_params)
        elif self.method == "global":
            flat_shape = flat_x.shape
            flat_x = self.transformer.transform(
                flat_x.reshape((-1, 1)), **fit_params
            ).reshape(flat_shape)
        return flat_to_inst(flat_x, slices)
