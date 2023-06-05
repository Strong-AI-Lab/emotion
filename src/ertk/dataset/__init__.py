"""
Dataset classes and functions
=============================

This module contains classes and functions for working with datasets.


Features
--------
.. autosummary::
    :toctree: generated/

    read_features
    read_features_iterable
    register_format
    write_features


Annotations
-----------
.. autosummary::
    :toctree: generated/

    read_annotations
    write_annotations


Dataset loading and manipulation
--------------------------------
.. autosummary::
    :toctree: generated/

    CombinedDataset
    CorpusInfo
    DataLoadConfig
    DataSelector
    Dataset
    DatasetConfig
    MapGroups
    RemoveGroups
    SubsetInfo
    load_datasets_config
    load_multiple


Utilities
---------
.. autosummary::
    :toctree: generated/

    get_audio_paths
    resample_audio
    resample_rename_clips
    write_filelist
"""

from .annotation import read_annotations, write_annotations
from .dataset import (
    CombinedDataset,
    CorpusInfo,
    DataLoadConfig,
    DataSelector,
    Dataset,
    DatasetConfig,
    MapGroups,
    RemoveGroups,
    SubsetInfo,
    load_datasets_config,
    load_multiple,
)
from .features import (
    read_features,
    read_features_iterable,
    register_format,
    write_features,
)
from .utils import (
    get_audio_paths,
    resample_audio,
    resample_rename_clips,
    write_filelist,
)
