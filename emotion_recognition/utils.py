from typing import List, Tuple, Union
import numpy as np


def pad_arrays(arrays: Union[List[np.ndarray], np.ndarray], pad: int = 32):
    """Pads each array to the nearest multiple of `pad` greater than the
    array size. Assumes axis 0 of each sub-array, or axis 1 of x is
    time.
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
    """Clips each array to the specified maximum length."""
    for i in range(len(arrays)):
        arrays[i] = np.copy(arrays[i][:length])
    assert all(len(x) <= length for x in arrays)
    return arrays


def shuffle_multiple(*arrays, numpy_indexing: bool = True):
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

    perm = np.random.permutation(len(arrays[0]))
    new_arrays = [array[perm] if numpy_indexing else [array[i] for i in perm]
                  for array in arrays]
    return new_arrays


def batch_arrays(arrays_x: List[np.ndarray], y: np.ndarray,
                 batch_size: int = 32, shuffle: bool = True,
                 uniform_batch_size: bool = False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Batches a list of arrays of different sizes, grouping them by
    size. This is designed for use with variable length sequences. Each
    batch will have a maximum of batch_size arrays, but may have less if
    there are fewer arrays of the same length. It is recommended to use
    the pad_arrays() method of the LabelledDataset instance before using
    this function, in order to quantise the lengths somewhat.

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
        y_list[i] has the same size as x_list[i].
    """
    if shuffle:
        arrays_x, y = shuffle_multiple(arrays_x, y, numpy_indexing=False)

    fixed_shape = arrays_x[0].shape[1:]
    lengths = [x.shape[0] for x in arrays_x]
    unique_len = np.unique(lengths)

    x_list = []
    y_list = []
    for length in unique_len:
        idx = np.nonzero(lengths == length)[0]
        for b in range(0, len(idx), batch_size):
            batch_idx = idx[b:b + batch_size]
            size = batch_size if uniform_batch_size else len(batch_idx)
            _x = np.zeros((size, length) + fixed_shape, dtype=np.float32)
            _y = np.zeros(size, dtype=np.float32)
            _y[:size] = y[batch_idx]
            for i, j in enumerate(batch_idx):
                _x[i, ...] = arrays_x[j]
            x_list.append(_x)
            y_list.append(_y)
    x_list = np.array(x_list, dtype=object)
    y_list = np.array(y_list,
                      dtype=np.float32 if uniform_batch_size else object)
    return x_list, y_list
