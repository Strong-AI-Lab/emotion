"""Various utility functions for modifying arrays and other things."""

from os import PathLike
from pathlib import Path
from typing import (
    Callable,
    Container,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import click
import numpy as np

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
    array size. Assumes axis 0 of each sub-array, or axis 1 of x is
    time.

    NOTE: This function modifies the arrays in-place.
    """
    if isinstance(arrays, np.ndarray) and len(arrays.shape) > 1:
        # Pad axis 1
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
