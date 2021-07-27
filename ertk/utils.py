"""Various utility functions for modifying arrays and other things."""

from os import PathLike
from pathlib import Path
from typing import (
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
from sklearn.base import TransformerMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    LeaveOneOut,
)

__all__ = [
    "PathOrStr",
    "PathlibPath",
    "itmap",
    "ordered_intersect",
    "frame_arrays",
    "pad_arrays",
    "clip_arrays",
    "transpose_time",
    "shuffle_multiple",
    "batch_arrays",
]

PathOrStr = Union[PathLike, str]

ScoreFunction = Callable[[np.ndarray, np.ndarray], float]


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
        self.tqdm_args = {"desc": desc, "unit": unit, "leave": leave}
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
        if isinstance(x, list):
            return list(s(y) for y in x)
        elif isinstance(x, tuple):
            return tuple(s(y) for y in x)
        else:
            return s(x)

    return _map


def ordered_intersect(a: Iterable, b: Container) -> List:
    """Returns a list of the intersection of `a` and `b`, in the order
    elements appear in `a`.
    """
    return [x for x in a if x in b]


def flat_to_inst(x: np.ndarray, slices: np.ndarray) -> np.ndarray:
    """Takes a flattened 2D data array and converts it to either a
    contiguous 2D/3D array or a variable-length 3D array, with one
    feature vector/matrix per instance.
    """

    if len(x) == len(slices):
        # 2-D contiguous array
        return x
    elif all(x == slices[0] for x in slices):
        # 3-D contiguous array
        assert len(x) % len(slices) == 0
        seq_len = len(x) // len(slices)
        return np.reshape(x, (len(slices), seq_len, x[0].shape[-1]))
    else:
        # 3-D variable length array
        indices = np.cumsum(slices)
        arrs = np.split(x, indices[:-1], axis=0)
        return np.array(arrs, dtype=object)


def inst_to_flat(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The inverse of flat_to_inst(). Takes an instance matrix and
    converts to a "flattened" 2D matrix.
    """

    slices = np.ones(len(x), dtype=int)
    if len(x.shape) != 2:
        slices = np.array([len(_x) for _x in x])
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

    NOTE: This function modifies the arrays in-place.
    """
    if isinstance(arrays, np.ndarray) and len(arrays.shape) > 1:
        padding = int(np.ceil(arrays.shape[1] / pad)) * pad - arrays.shape[1]
        extra_dims = tuple((0, 0) for _ in arrays.shape[2:])
        arrays = np.pad(arrays, ((0, 0), (0, padding)) + extra_dims)
    else:
        for i in range(len(arrays)):
            x = arrays[i]
            padding = int(np.ceil(x.shape[0] / pad)) * pad - x.shape[0]
            arrays[i] = np.pad(x, ((0, padding), (0, 0)))
    return arrays


def clip_arrays(arrays: Union[List[np.ndarray], np.ndarray], length: int):
    """Clips each array to the specified maximum length.

    NOTE: This function modifies the arrays in-place.
    """
    for i in range(len(arrays)):
        arrays[i] = np.copy(arrays[i][:length])
    assert all(len(x) <= length for x in arrays)
    return arrays


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


def shuffle_multiple(*arrays: np.ndarray, numpy_indexing: bool = True):
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
    """Cross-validation method that uses the training set as validation
    set.
    """

    def split(self, X, y, groups):
        return np.arange(len(X)), np.arange(len(X))

    def get_n_splits(self, X, y, groups):
        return 1


def get_cv_splitter(group: bool, k: int, test_size: float = 0.2):
    if k == 0:
        return TrainValidation()
    if group:
        if k > 1:
            return GroupKFold(k)
        elif k == 1:
            return GroupShuffleSplit(1, test_size=test_size)
        return LeaveOneGroupOut()
    if k > 1:
        return StratifiedKFold(k, shuffle=True)
    elif k == 1:
        return StratifiedShuffleSplit(1, test_size=test_size)
    return LeaveOneOut()


def group_transform(
    x: np.ndarray,
    groups: np.ndarray,
    transform: TransformerMixin,
    *,
    inplace: bool = False,
):
    """Per-group ("online") transformation (e.g. standardisation).

    Args:
    -----
    x: np.ndarray
        The data matrix to transform.
    groups: np.ndarray
        Groups assignment for each instance. It must be the case that
        len(groups) == len(x).
    transform:
        The transformation to apply. Must implement fit_transform().
    inplace: bool
        Whether to modify x in-place. Default is False so that a copy is
        made.

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
        flat = transform.fit_transform(flat)
        x[groups == g_id] = flat_to_inst(flat, slices)
    return x


def get_scores(
    scoring: Union[str, List[str], Dict[str, ScoreFunction], Callable[..., float]],
    y_pred,
    y_true,
):
    from sklearn.model_selection._validation import _score

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

    dummy = DummyEstimator(y_pred)
    scores = {}
    if isinstance(scoring, str):
        val = get_scorer(scoring)(dummy, None, y_true)
        scores["test_score"] = val
        scores["test_" + scoring] = val
    elif isinstance(scoring, (list, dict)):
        if isinstance(scoring, list):
            scoring = {x: get_scorer(x) for x in scoring}
        _scores = _score(dummy, None, y_true, scoring)
        for k, v in _scores.items():
            scores["test_" + k] = v
    elif callable(scoring):
        scores["test_score"] = scoring(dummy, None, y_true)
    return scores
