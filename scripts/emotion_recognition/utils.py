import numpy as np


def shuffle_multiple(*arrays, numpy_indexing: bool = True):
    """Shuffles multiple arrays or lists in sync. Useful for shuffling the data
    and labels in a dataset separately while keeping them synchronised.

    Parameters:
    -----------
    arrays, iterable of array-like
        The arrays to shuffle. They must all have the same size of first
        dimension.
    numpy_indexing: bool, default = True
        Whether or not to use NumPy-style indexing or list comprehension.

    Returns:
    shuffled_arrays: iterable of array-like
        The shuffled arrays.
    """
    if any(len(arrays[0]) != len(x) for x in arrays):
        raise ValueError("Not all arrays have the same size.")

    perm = np.random.permutation(len(arrays[0]))
    new_arrays = [array[perm] if numpy_indexing else [array[i] for i in perm]
                  for array in arrays]
    return new_arrays
