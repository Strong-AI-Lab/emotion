"""Transforms to use in estimators.

.. autosummary::
    :toctree: generated/

    group_transform
    instance_transform
    GroupTransformWrapper
    InstanceTransformWrapper
    SequenceTransform
    SequenceTransformWrapper
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ertk.utils import flat_to_inst, inst_to_flat

__all__ = [
    "group_transform",
    "instance_transform",
    "GroupTransformWrapper",
    "InstanceTransformWrapper",
    "SequenceTransform",
    "SequenceTransformWrapper",
]


def group_transform(
    x: np.ndarray,
    groups: np.ndarray,
    transform: TransformerMixin,
    *,
    inplace: bool = False,
    **fit_params,
):
    """Per-group (offline) transformation (e.g. standardisation).

    Parameters
    ----------
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

    Returns
    -------
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

    Parameters
    ----------
    x: np.ndarray
        The data matrix to transform. Each x[i] must be a 2D instance.
    transform:
        The transformation to apply. Must implement fit_transform().
    inplace: bool
        Whether to modify x in-place. Default is False so that a copy is
        made.
    **fit_params:
        Other keyword arguments to pass to the transform.fit() method.

    Returns
    -------
    x: np.ndarray
        The modified data matrix with transformations applied to each
        instance individually.
    """
    return group_transform(
        x, np.arange(len(x)), transform, inplace=inplace, **fit_params
    )


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
        return instance_transform(X, self.transformer, inplace=False, **fit_params)


class SequenceTransform(TransformerMixin, BaseEstimator):
    """Transform designed to process sequences of vectors."""

    pass


class SequenceTransformWrapper(SequenceTransform):
    """Wrapper around a scikit-learn transform that can process
    sequences of vectors.

    Parameters
    ----------
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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(transformer={self.transformer}, "
            f"method={self.method})"
        )
