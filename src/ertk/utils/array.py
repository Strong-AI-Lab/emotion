from typing import List, Optional, Sequence, Tuple, Union, overload

import numpy as np


def make_array_array(x: List[np.ndarray]) -> np.ndarray:
    """Helper function to make an array of arrays. The returned array
    has `shape==(len(x),)` and `dtype==object`.
    """
    arr = np.empty(len(x), dtype=object)
    for i, a in enumerate(x):
        arr[i] = a
    return arr


def check_3d(arrays: Union[Sequence[np.ndarray], np.ndarray]) -> None:
    """Checks if an array is 3D or each array in a list is 2D. Raises an
    exception if this isn't the case.
    """
    if any(len(x.shape) != 2 for x in arrays):
        raise ValueError("arrays must be 3D (contiguous or vlen).")


def frame_array(
    x: np.ndarray,
    frame_size: int,
    frame_shift: int,
    pad: bool = False,
    axis: int = 0,
    copy: bool = True,
) -> np.ndarray:
    """Frames an array over a given axis with optional padding and
    copying.

    Parameters:
    -----------
    x: np.ndarray
        Array to frame.
    frame_size: int
        Width of frame.
    frame_shift: int
        Amount each window is moved for subsequent frames.
    pad: bool
        If the frames don't neatly fit the axis length, this indicated
        whether to pad or cut the remaining length. Default is `False`
        which means any remaining length is cut. If `pad=True` then any
        remaining length is padded with zeros and an additional frame is
        produced. If frames neatly fit into the axis length then this
        option has no effect.
    axis: int
        Axis to frame over. Default is 0.
    copy: bool
        Whether to copy and return a contiguous array (more memory
        usage). Default is `True` so that a contiguous copy is retured.
        Note: if `copy=False` it is assumed that the input array is
        contiguous so that a view with modified strides can be properly
        created.

    Returns:
    --------
    frames: np.ndarray
        The resulting frames. If `copy=False` then these are read-only
        views into x, otherwise a contiguous array is returned.
    """
    idx_slice = tuple(slice(None) for _ in range(axis))

    num_frames = (x.shape[axis] - frame_size) // frame_shift + 1
    if num_frames <= 0:
        if not pad:
            raise ValueError(
                "The length of the sequence is shorter than frame_size, but pad=False, "
                "so no frames will be generated."
            )
        num_frames = 0
    remainder = (x.shape[axis] - frame_size) % frame_shift
    if remainder != 0:
        num_frames += 1 if pad else 0
    # Insert new dim before axis with num_frames, and axis is replaced
    # with the size of each frame.
    new_shape = x.shape[:axis] + (num_frames, frame_size) + x.shape[axis + 1 :]

    if copy:
        out = np.zeros(new_shape, dtype=x.dtype)
        for i in range(0, x.shape[axis] - frame_size + 1, frame_shift):
            out[idx_slice + (i // frame_shift,)] = x[
                idx_slice + (slice(i, i + frame_size),)
            ]
        if remainder != 0 and pad:
            if num_frames == 1:
                left = x.shape[axis]
            else:
                left = frame_size + remainder - frame_shift
            # out[..., -1, 0:left, ...] = x[..., -left:, ...]
            out[idx_slice + (-1, slice(0, left))] = x[idx_slice + (slice(-left, None),)]

        return out

    from numpy.lib.stride_tricks import as_strided

    if remainder != 0:
        if pad:
            # Padding creates a new array copy
            raise ValueError("pad must be False when copy=False")
        # x[:, :, ..., 0:N-r, :, ...]
        x = x[idx_slice + (slice(0, x.shape[axis] - remainder),)]
    # Add a new stride for the num_frames part of shape which moves by
    # frame_shift for each frame. The stride within each frame is the
    # same as for the original axis.
    new_strides = x.strides[:axis] + (frame_shift * x.strides[axis],) + x.strides[axis:]
    return as_strided(x, new_shape, new_strides, writeable=False)


def frame_arrays(
    arrays: Union[List[np.ndarray], np.ndarray],
    frame_size: int,
    frame_shift: int,
    max_frames: Optional[int] = None,
    vlen: bool = False,
) -> np.ndarray:
    """Creates sequences of frames from the given arrays. This is mainly
    a convenience wrapper around `frame_array()` for instance arrays.

    Parameters:
    -----------
    arrays: list of np.ndarray
        The arrays to process.
    frame_size: int
        Length of each frame.
    frame_shift: int
        Amount each frame is shifted.
    max_frames: int, optional
        If given, only the first max_frames are returned for each array.
    vlen: bool
        Whether to return a contiguous array if `vlen=False` (default)
        or an array of variable length arrays.

    Returns:
    --------
    framed_arrays:
        Framed arrays.
    """
    check_3d(arrays)

    if max_frames is None:
        max_len = max(len(x) for x in arrays)
        max_frames = (max_len - frame_size) // frame_shift + 1

    # Generator
    arrs = (
        frame_array(x, frame_size, frame_shift, axis=0, pad=True, copy=True)[
            :max_frames
        ]
        for x in arrays
    )
    if vlen:
        return make_array_array(list(arrs))

    n_features = arrays[0].shape[1]
    framed_arrays = np.zeros(
        (len(arrays), max_frames, frame_size, n_features), dtype=arrays[0].dtype
    )
    for i, arr in enumerate(arrs):
        framed_arrays[i, : len(arr)] = arr
    return framed_arrays


def pad_array(
    x: np.ndarray, to_multiple: int = None, to_size: int = None, axis: int = 0
):
    """Pads an array either to a multiple of `to_multiple` or to the
    exact length `to_size` along `axis`.


    Parameters:
    -----------
    x: np.ndarray
        The array to pad.
    to_multiple: int, optional
        If given, `x` is padded so that it's length is the nearest
        larger multiple of `to_multiple`.
    to_size: int, optional
        If given, `x` is padded so that it's length is exactly
        `to_size`. Default is None, and instead `to_multiple` is used.
        Exactly one of `to_size` and `to_multiple` must be given.
    axis: int
        The axis along which to pad. Default is axis 0.

    Returns:
    --------
    padded: np.ndarray
        The padded array.
    """
    if to_multiple is not None and to_size is not None:
        raise ValueError("Only one of `to_multiple` and `to_size` should be given.")

    if to_size is not None:
        to_pad = to_size - x.shape[axis]
        if to_pad < 0:
            raise ValueError("The length of `x` is already greater than `to_size`.")
    elif to_multiple is not None:
        to_pad = int(np.ceil(x.shape[axis] / to_multiple)) * to_multiple - x.shape[axis]
    else:
        raise ValueError("One of `to_multiple` and `to_size` must be given.")

    if to_pad == 0:
        return x.copy()  # A copy is expected when padding

    pad = [(0, 0) for _ in x.shape]
    pad[axis] = (0, to_pad)
    return np.pad(x, pad)


@overload
def pad_arrays(arrays: List[np.ndarray], pad: int) -> List[np.ndarray]:
    pass


@overload
def pad_arrays(arrays: np.ndarray, pad: int) -> np.ndarray:
    pass


def pad_arrays(arrays: Union[List[np.ndarray], np.ndarray], pad: int = 32):
    """Pads each array to the nearest multiple of `pad` greater than the
    array size. Assumes axis 0 of each sub-array, or axis 1 of x, is
    the time axis. This is mainly a wrapper around `pad_array()` for
    instance arrays.
    """
    if isinstance(arrays, np.ndarray) and len(arrays.shape) > 1:
        # Contiguous 2D/3D
        return pad_array(arrays, to_multiple=pad, axis=1)
    new_arrays = [pad_array(x, to_multiple=pad, axis=0) for x in arrays]
    if isinstance(arrays, np.ndarray):
        if all(x.shape == new_arrays[0].shape for x in new_arrays):
            # Contiguous array
            return np.stack(new_arrays)
        # Array of variable-length arrays
        return make_array_array(new_arrays)
    # List
    return new_arrays


@overload
def clip_arrays(arrays: List[np.ndarray], length: int, copy: bool) -> List[np.ndarray]:
    pass


@overload
def clip_arrays(arrays: np.ndarray, length: int, copy: bool) -> np.ndarray:
    pass


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
        return make_array_array(new_arrays)
    return [x[:length].copy() if copy else x[:length] for x in arrays]


@overload
def transpose_time(arrays: List[np.ndarray]) -> List[np.ndarray]:
    pass


@overload
def transpose_time(arrays: np.ndarray) -> np.ndarray:
    pass


def transpose_time(arrays: Union[List[np.ndarray], np.ndarray]):
    """Transpose the time and feature axis of each array. Requires each
    array be 2-D.

    Note: This function modifies the arrays in-place.
    """
    check_3d(arrays)
    if isinstance(arrays, np.ndarray) and len(arrays.shape) == 3:
        arrays = arrays.transpose(0, 2, 1)
    else:
        for i in range(len(arrays)):
            arrays[i] = arrays[i].transpose()
    assert all(x.shape[0] == arrays[0].shape[0] for x in arrays)
    return arrays


def shuffle_multiple(
    *arrays: Union[np.ndarray, Sequence],
    numpy_indexing: bool = True,
    seed: Optional[int] = None,
):
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

    perm = np.random.default_rng(seed).permutation(len(arrays[0]))
    new_arrays = [
        array[perm] if numpy_indexing else [array[i] for i in perm] for array in arrays
    ]
    return new_arrays


def batch_arrays_by_length(
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
    if shuffle:
        arrays_x, y = shuffle_multiple(arrays_x, y, numpy_indexing=False)

    # TODO: implement batch generator

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
    x_batch = make_array_array(xlist)
    if uniform_batch_size:
        y_batch = np.array(ylist, dtype=y_dtype)
    else:
        y_batch = make_array_array(ylist)
    return x_batch, y_batch


def flat_to_inst(x: np.ndarray, slices: Union[np.ndarray, List[int]]) -> np.ndarray:
    """Takes a concatenated 2D data array and converts it to either a
    contiguous 2D/3D array or a variable-length 3D array, with one
    feature vector/matrix per instance.

    Parameters:
    -----------
    x: numpy.ndarray
        Contiguous 2D array containing concatenated instance data.
    slices: list or numpy.ndarray
        Array containing lengths of sequences in x corresponding to
        instance data.

    Returns:
    --------
    inst: numpy.ndarray
        Instance array, which is either contiguous 2D, contiguous 3D or
        an array of 2D arrays with varying lengths. It should be the
        case that `inst[i] == x[s : s + slices[i]]`, where `s =
        sum(slices[:i])`.
    """

    if len(x) == len(slices):
        # 2D contiguous array
        return x
    elif all(x == slices[0] for x in slices):
        # 3D contiguous array
        assert len(x) % len(slices) == 0
        return x.reshape(len(slices), len(x) // len(slices), x[0].shape[-1])
    else:
        # 3D variable length array
        start_idx = np.cumsum(slices)[:-1]
        return make_array_array(np.split(x, start_idx, axis=0))


def inst_to_flat(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The inverse of flat_to_inst(). Takes an instance 'matrix' and
    converts to a "flattened" 2D matrix.

    Parameters:
    -----------
    x: numpy.ndarray
        The instance matrix, which is either a contiguous 2D or 3D
        array, or an array of 1D or 2D arrays.

    Returns:
    --------
    flat: numpy.ndarray
        The contiguous 2D matrix containing data for all instances.
    slices: numpy.ndarray
        Array of slices (sequence lengths) in `flat`. It should be the
        case that `x[i] == flat[s : s + slices[i]]`, where `s =
        sum(slices[:i])`.
    """

    slices = np.ones(len(x), dtype=int)
    if len(x.shape) != 2:
        if x[0].ndim == 1:
            # Array of 1D arrays
            x = np.stack(x)  # type: ignore
        else:
            slices = np.array([len(_x) for _x in x])
            if len(x.shape) == 3:
                # Contiguous 3D array
                x = x.reshape(sum(slices), x.shape[2])
            elif x[0].base is not None and all(_x.base is x[0].base for _x in x):
                # Array of views into contiguous array
                x = x[0].base
            else:
                # Array of separate arrays
                x = np.concatenate(x)
    assert sum(slices) == len(x)
    return x, slices
