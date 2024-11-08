"""
Utility functions
=================

This module contains miscellaneous utility functions.


Array utilities
---------------
.. autosummary::
    :toctree: generated/

    batch_arrays_by_length
    clip_arrays
    flat_to_inst
    frame_array
    frame_arrays
    inst_to_flat
    is_mono_audio
    make_array_array
    pad_arrays
    shuffle_multiple
    transpose_time


Generic utilities
-----------------
.. autosummary::
    :toctree: generated/

    batch_iterable
    filter_kwargs
    itmap
    ordered_intersect
    ordered_subsets
    subsets


Miscellaneous utilities
-----------------------
.. autosummary::
    :toctree: generated/

    TqdmMultiprocessing
    TqdmParallel


type definitions
----------------
.. autosummary::
    :toctree: generated/

    PathOrStr
    ScoreFunction
"""

from .array import (
    batch_arrays_by_length,
    clip_arrays,
    flat_to_inst,
    frame_array,
    frame_arrays,
    inst_to_flat,
    is_mono_audio,
    make_array_array,
    pad_arrays,
    shuffle_multiple,
    transpose_time,
)
from .generic import (
    batch_iterable,
    filter_kwargs,
    itmap,
    ordered_intersect,
    ordered_subsets,
    subsets,
)
from .misc import TqdmMultiprocessing, TqdmParallel
from .types import PathOrStr, ScoreFunction

__all__ = [
    "batch_arrays_by_length",
    "clip_arrays",
    "flat_to_inst",
    "frame_array",
    "frame_arrays",
    "inst_to_flat",
    "is_mono_audio",
    "make_array_array",
    "pad_arrays",
    "shuffle_multiple",
    "transpose_time",
    "batch_iterable",
    "filter_kwargs",
    "itmap",
    "ordered_intersect",
    "ordered_subsets",
    "subsets",
    "TqdmMultiprocessing",
    "TqdmParallel",
    "PathOrStr",
    "ScoreFunction",
]
