"""Dataset classes and related functions."""

import copy
import logging
import os
import warnings
from collections.abc import Collection, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from functools import partial, reduce
from itertools import chain
from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
from numpy.lib.arraysetops import setdiff1d
from omegaconf import MISSING, OmegaConf
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

from ertk.config import ERTKConfig, get_arg_mapping
from ertk.dataset.annotation import read_annotations
from ertk.dataset.features import find_features_file, read_features
from ertk.dataset.utils import get_audio_paths
from ertk.transform import group_transform
from ertk.utils import PathOrStr, clip_arrays, frame_arrays, pad_arrays, transpose_time

__all__ = [
    "Dataset",
    "CombinedDataset",
    "DatasetConfig",
    "DataLoadConfig",
    "DataSelector",
    "SubsetInfo",
    "MapGroups",
    "RemoveGroups",
    "CorpusInfo",
    "load_multiple",
    "load_datasets_config",
]

logger = logging.getLogger(__name__)


@dataclass
class MapGroups:
    """Defined a mapping between categorical groups."""

    map: dict[str, str] = MISSING


@dataclass
class RemoveGroups:
    """Defines a set of groups to keep or remove from a dataset."""

    drop: list[str] = field(default_factory=list)
    keep: list[str] = field(default_factory=list)


@dataclass
class SubsetInfo:
    """Defines a subset of a dataset."""

    clips: str = MISSING
    description: str = ""


@dataclass
class DataSelector(ERTKConfig):
    """Defines a selection of data to keep."""

    subset: str = ""
    groups: dict[str, RemoveGroups] = field(default_factory=dict)


@dataclass
class CorpusInfo(ERTKConfig):
    """Defines information about a dataset."""

    name: str = MISSING
    description: str = ""
    annotations: list[str] = field(default_factory=list)
    partitions: list[str] = field(default_factory=list)
    ratings: list[str] = field(default_factory=list)
    subsets: dict[str, SubsetInfo] = field(default_factory=dict)
    default_subset: str = "all"
    features_dir: str = "features"


@dataclass
class DatasetConfig(ERTKConfig):
    """Defines a dataset configuration."""

    path: str = MISSING
    features: Optional[str] = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    subset: str = "default"
    map_groups: dict[str, MapGroups] = field(default_factory=dict)
    remove_groups: dict[str, RemoveGroups] = field(default_factory=dict)
    rename_annotations: dict[str, str] = field(default_factory=dict)
    select: Optional[DataSelector] = None
    clip_seq: Optional[int] = None
    pad_seq: Optional[int] = None


@dataclass
class DataLoadConfig(ERTKConfig):
    """Defines a configuration for loading one or more datasets."""

    datasets: dict[str, DatasetConfig] = MISSING
    features: Optional[str] = None
    label: Optional[str] = None
    map_groups: dict[str, MapGroups] = field(default_factory=dict)
    remove_groups: dict[str, RemoveGroups] = field(default_factory=dict)
    select: Optional[DataSelector] = None
    clip_seq: Optional[int] = None
    pad_seq: Optional[int] = None


_AUDIO_PATH_KEY = "_audio_path"


class Dataset:
    """Class representing a generic dataset, consisting of a set of
    features and optional partitions and annotations. Has various
    preprocessing methods.

    An annotation is a scalar value for all instances in the dataset.

    A partition is a partition of instances into disjoint groups (e.g.
    speakers). A partition should be complete in that each instance is
    in exactly one group in the partition. Each partition has a
    corresponding annotation with the same name, using the group names
    as categorical annotations.

    Parameters
    ----------
    corpus_info: Pathlike or str, optional
        Path to corpus info in YAML format.
    features: Pathlike or str, optional
        Path to features file, or unique name of features in corpus
        features directory.
    subset: str, optional
        The subset of instances to use.
    """

    _annotations: pd.DataFrame
    _corpus: str = ""
    _feature_names: list[str]
    _names: pd.Index
    _partitions: Set[str]
    _subset: str = ""
    _subsets: dict[str, list[str]]
    _x: np.ndarray
    _label_annot: str = ""
    _ratings: dict[str, pd.DataFrame]
    _description: str = ""

    # "Private" vars
    _default_subset: str = ""
    _features: str = ""
    _features_path: Optional[Path] = None
    _features_dir: Path = Path()
    _subset_paths: dict[str, Path]

    def __init__(
        self,
        corpus_info: PathOrStr,
        features: Optional[PathOrStr] = None,
        subset: str = "default",
        label: str = "label",
    ):
        self._init()
        self.init_corpus_info(corpus_info)
        self.use_subset(subset)
        if features is not None:
            self.update_features(features)
        self._label_annot = label

    def _init(self) -> None:
        self._annotations = pd.DataFrame()
        self._feature_names = []
        self._names = pd.Index([])
        self._partitions = set()
        self._subsets = {}
        self._subset_paths = {}
        self._x = np.empty((0, 0), dtype=np.float32)
        self._ratings = {}

    def _copy_from_dataset(self, other: "Dataset") -> None:
        self._annotations = other._annotations.copy()
        self._corpus = other._corpus
        self._default_subset = other._default_subset
        self._feature_names = other._feature_names.copy()
        self._features = other._features
        self._features_path = other._features_path
        self._features_dir = other._features_dir
        self._names = other._names.copy()
        self._partitions = other._partitions.copy()
        self._subset = other._subset
        self._subset_paths = other._subset_paths.copy()
        self._subsets = copy.deepcopy(other._subsets)
        self._x = other._x.copy()
        self._label_annot = other._label_annot
        self._ratings = copy.deepcopy(other._ratings)
        self._description = other._description
        if len(self._x.shape) == 1:
            # Non-contiguous array, so copy each contiguous sub-array
            for i in range(len(self._x)):
                self._x[i] = self._x[i].copy()

    def init_corpus_info(self, path: PathOrStr) -> None:
        """Initialise corpus metadata from YAML.

        Parameters
        ----------
        path: os.Pathlike or str
            The path to a YAML file containing corpus metadata.
        """
        logger.debug(f"Initialising corpus from {path}")

        path = Path(path)
        corpus_info = CorpusInfo.from_file(path)
        self._corpus = corpus_info.name
        self._description = corpus_info.description
        self._features_dir = path.parent / corpus_info.features_dir
        self._default_subset = corpus_info.default_subset
        for subset, subset_info in corpus_info.subsets.items():
            clips_file = path.parent / subset_info.clips
            if clips_file.suffix == "":
                clips_file = clips_file.with_suffix(".txt")
            self._subset_paths[subset] = clips_file
            name_to_path = {x.stem: str(x) for x in get_audio_paths(clips_file)}
            self.subsets[subset] = list(name_to_path.keys())
            if subset == "all":
                self._names = pd.Index(self.subsets["all"])
                # This will initialise the index of self.annotations
                self._annotations.index = pd.Index(self.subsets["all"])
                self.update_annotation(_AUDIO_PATH_KEY, list(name_to_path.values()))
        for partition in corpus_info.partitions:
            self.update_annotation(
                partition, path.parent / f"{partition}.csv", dtype="category"
            )
        for annotation in corpus_info.annotations:
            self.update_annotation(annotation, path.parent / f"{annotation}.csv")
        for ratings in corpus_info.ratings:
            self.update_ratings(ratings, path.parent / f"{ratings}.csv")
        self.update_annotation(
            "corpus", [self.corpus] * len(self._names), dtype="category"
        )

    def _verify_index(
        self, df_or_series: Union[pd.DataFrame, pd.Series], msg: Optional[str] = None
    ):
        msg = msg or "Incomplete index"
        try:
            df_or_series.loc[self.names]
        except KeyError as e:
            raise RuntimeError(msg) from e

    def _verify_annotations(self, msg: Optional[str] = "Incomplete annotation"):
        """Make sure there is a value for each instance for each annotation."""
        self._verify_index(self.annotations, msg=msg)

    def _verify_ratings(self):
        for name, df in self.ratings.items():
            try:
                self._verify_index(df, msg=f"Incomplete ratings for '{name}'")
            except RuntimeError as e:
                warnings.warn(str(e), RuntimeWarning)

    def _reset_categorical(self):
        """Make sure there are no extraneous categories in any
        partition.
        """
        for part in self.partitions:
            new = self._annotations.loc[self.names, part].cat.remove_unused_categories()
            self._annotations[part] = new

    def update_features(self, features: PathOrStr, **read_kwargs) -> None:
        """Update the features matrix and feature names for this dataset.

        Parameters
        ----------
        features: os.PathLike or str
            Path to a set of features or unique name of features in
            corpus features dir.
        **read_kwargs:
            Other arguments to pass to `read_features()`.
        """
        if features == "raw_audio":
            self._features = "raw_audio"
            features = self._subset_paths[self.subset or "all"]
        else:
            features = Path(features)
            if not features.exists():
                try:
                    features = find_features_file(
                        self._features_dir.glob(f"{features}.*")
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"Cannot find features '{features}' in directory "
                        f"'{self._features_dir}'"
                    ) from e
            self._features = features.stem
        self._features_path = features
        data = read_features(features, **read_kwargs)
        self._feature_names = data.feature_names

        assert len(self.names) > 0
        missing = self.names.difference(data.names)
        if len(missing) > 0:
            raise ValueError(
                f"Features at {features} don't contain all instances, these are"
                f"missing: {missing}"
            )
        # To avoid reordering the feature matrix, reorder self.names instead
        data_names = pd.Index(data.names)
        idx = data_names.isin(self.names)
        self._x = data.features[idx]
        self._names = data_names[idx]

    def use_subset(self, subset: str = "default") -> None:
        """Use a different subset of the instances.

        Parameters
        ----------
        subset: str
            Name of subset to use. Default is "default" which uses the
            default subset specified in corpus_info.
        """
        if subset == "default":
            subset = self._default_subset
        self._subset = subset
        self._names = pd.Index(self.subsets[subset])
        self._reset_categorical()
        self._verify_annotations()

    def get_idx_for_names(self, names: Collection[str]) -> np.ndarray:
        """Gets indices of instances corresponding to `names`.

        Parameters
        ----------
        names: collection of str
            The names to get indices for.

        Returns
        -------
        idx: np.ndarray
            The indices corresponding to `names`, in order.
        """
        return np.flatnonzero(self.names.isin(names))

    @overload
    def get_idx_for_split(
        self,
        split: Union[str, dict[str, Collection[str]], DataSelector],
        return_complement: Literal[False] = False,
    ) -> np.ndarray: ...

    @overload
    def get_idx_for_split(
        self,
        split: Union[str, dict[str, Collection[str]], DataSelector],
        return_complement: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def get_idx_for_split(
        self,
        split: Union[str, dict[str, Collection[str]], DataSelector],
        return_complement: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Gets indices of instances corresponding to the selection
        given by `split`.

        Parameters
        ----------
        split: str or dict
            Either a string containing the subset to select, groups to
            select, a path to such a config file, or a mapping object
            containing the groups to select.
        return_complement: bool
            If True, return a tuple of `idx`, `comp_idx` where
            `comp_idx` contains the complement indices not in the split.

        Returns
        -------
        idx: np.ndarray
            The corresponding indices.
        comp_idx: np.ndarray, optional
            If `return_complement` is True, this is also returned and
            contains the complement indices.
        """
        subset_idx = None
        if isinstance(split, str):
            if ":" not in split and not Path(split).exists():  # is a subset
                return self.get_idx_for_names(self.subsets[split])
            mapping = get_arg_mapping(split)
        elif isinstance(split, dict):
            mapping = split
        else:
            if OmegaConf.get_type(split) is None:
                raise TypeError("Config is of incorrect type")
            if split.subset:
                subset_idx = self.get_idx_for_names(self.subsets[split.subset])
            mapping = {
                k: v.keep if v.keep else set(self.get_group_names(k)) - set(v.drop)
                for k, v in split.groups.items()
            }

        indices_parts: dict[str, np.ndarray] = {}
        for part_name in mapping:
            sel = mapping[part_name]
            sel = [sel] if isinstance(sel, str) else sel
            group_names = self.get_group_names(part_name)
            sel_idx = [group_names.index(x) for x in sel]
            indices_parts[part_name] = np.flatnonzero(
                np.isin(self.get_group_indices(part_name), sel_idx)
            )

        all_idx = np.arange(len(self))
        indices = reduce(
            partial(np.intersect1d, assume_unique=True), indices_parts.values(), all_idx
        )
        if subset_idx is not None:
            indices = np.intersect1d(indices, subset_idx, assume_unique=True)
        comp_indices = setdiff1d(all_idx, indices, assume_unique=True)
        if return_complement:
            return np.sort(indices), np.sort(comp_indices)
        else:
            return np.sort(indices)

    def update_annotation(
        self,
        annot_name: str,
        annotations: Union[PathOrStr, Mapping[str, Any], Sequence[Any], pd.Series],
        dtype: Optional[Union[type, Literal["category"]]] = None,
    ) -> None:
        """Add or update an annotation.

        Parameters
        ----------
        annot_name: str
            The name of the annotation.
        annotations: PathLike, str, mapping, DataFrame, Series or sequence
            Annotations to add, similar to update_partition(). If
            PathLike or str, annotations are read from a CSV. If a dict,
            should be of the form {instance: annotation}. If a list,
            should have an annotation for each instance.
        dtype: type, optional
            The type of annotations for reading from CSV file. If the
            literal "category" is given the annotations are converted to
            the Pandas categorical dtype.
        """
        if isinstance(annotations, (os.PathLike, str)):
            series = read_annotations(annotations, dtype=dtype)
        elif isinstance(annotations, Mapping):
            series = pd.DataFrame.from_dict(annotations, orient="index")
            series = series.squeeze("columns")
        elif isinstance(annotations, pd.Series):
            series = annotations
        else:
            series = pd.Series(annotations, index=self.names)

        logger.debug(f"Updating annotation {annot_name} ({len(series)} values).")

        try:
            self._verify_index(series)
        except RuntimeError as e:
            raise RuntimeError(f"Annotation {annot_name} is incomplete") from e
        self._annotations[annot_name] = series
        if (
            dtype == "category"
            or (dtype is not None and issubclass(dtype, str))
            or series.dtype == object
            or series.dtype == pd.CategoricalDtype
        ):
            self.partitions.add(annot_name)
            self._annotations[annot_name] = self._annotations[annot_name].astype(
                "category"
            )

    def update_ratings(
        self,
        rating_set: str,
        ratings: Union[PathOrStr, Mapping[str, Mapping[str, Any]], pd.Series],
    ) -> None:
        """Update a set of ratings.

        Parameters
        ----------
        rating_set: str
            The name for this set of ratings.
        ratings: PathLike, str, mapping, DataFrame, Series
            The ratings to add. Must have a joint index where
        """
        if isinstance(ratings, (os.PathLike, str)):
            df = pd.read_csv(ratings, converters={0: str, 1: str})
            df = df.set_index(list(df.columns[0:2]))
        elif isinstance(ratings, pd.DataFrame):
            df = ratings
        elif isinstance(ratings, pd.Series):
            df = ratings.to_frame()

        self.ratings[rating_set] = df
        self._verify_ratings()

    def remove_ratings(self, rating_set: str) -> None:
        """Delete a set of ratings for this dataset.

        Parameters
        ----------
        rating_set: str
            The name of a set of ratings.
        """
        del self.ratings[rating_set]

    def map_groups(self, part_name: str, mapping: Mapping[str, str]) -> None:
        """Map group names in a partition.

        Parameters
        ----------
        part_name: str
            Name of partition.
        mapping: dict
            Group name mapping.
        """
        logger.debug(f"Mapping {part_name}: {mapping}")

        annot = self._annotations[part_name]
        try:
            new = annot.cat.rename_categories(mapping)
        except ValueError:
            # Not a 1-1 mapping
            _map = {x: x for x in annot.cat.categories}
            _map.update(mapping)
            new = annot.map(_map).astype("category")
        new = new.cat.reorder_categories(new.cat.categories.sort_values())
        self._annotations[part_name] = new

        # groups = self.annotations[part_name]
        # self.update_annotation(
        #     part_name, groups.map(lambda x: mapping.get(x, x)), dtype=str
        # )

    def remove_groups(
        self,
        part_name: str,
        *,
        drop: Collection[str] = None,
        keep: Collection[str] = None,
    ) -> None:
        """Remove instances corresponding to groups from the given
        partition.

        Parameters
        ----------
        part_name: str
            The partition name.
        drop: collection of str
            The groups to remove in given partition.
        keep: collection of str
            The groups to keep in given partition.
        """
        drop = set([] if drop is None else drop)
        keep = set([] if keep is None else keep)

        if (len(drop) == 0) == (len(keep) == 0):
            raise ValueError("Exactly one of drop and keep should be non-empty.")

        all_groups = set(self.get_group_names(part_name))
        unknown = drop.union(keep) - all_groups
        if len(unknown) > 0:
            raise ValueError(f"Cannot drop or keep {unknown} in partition {part_name}.")

        if len(keep) == 0:
            keep = all_groups - drop

        if len(drop) == 0:
            drop = all_groups - keep

        logger.debug(
            f"Dropping {len(drop)} and keeping {len(keep)} groups from "
            f"partition {part_name}."
        )

        new = self._annotations[part_name].cat.remove_categories(list(drop))
        self._annotations[part_name] = new

        keep_names = new.index[new.isin(keep)]
        self.remove_instances(keep=keep_names.intersection(self.names))

    def map_and_select(
        self,
        map: Mapping[str, Mapping[str, str]],
        select: Mapping[str, Union[str, Collection[str]]],
        remove: Mapping[str, Union[str, Collection[str]]],
    ) -> None:
        """Convenience function for mapping one or more partitions and
        then selecting one or more groups.

        Parameters
        ----------
        map: mapping
            The groups mapping. May have one or more partitions.
        select: mapping
            Mapping from partitions to groups to select.
        """
        for part in map:
            if part not in self.partitions:
                warnings.warn(
                    f"Partition {part} cannot be mapped as it is not in the dataset."
                )
                continue
            self.map_groups(part, map[part])
        for part in set(select).union(remove):
            if part not in self.partitions:
                warnings.warn(
                    f"Partition {part} cannot be selected/dropped as it is not in the "
                    "dataset."
                )
                continue
            keep = select.get(part, [])
            drop = remove.get(part, [])
            if isinstance(keep, str):
                keep = [keep]
            if isinstance(drop, str):
                drop = [drop]
            self.remove_groups(part, keep=keep, drop=drop)

    def rename_annotation(self, old_name: str, new_name: str) -> None:
        """Renames an annotation. This is useful for example if you want
        to select a different labelling to use. If an annotation already
        exists with the given new_name, then it is replaced
        destructively.

        Parameters
        ----------
        old_name: str
            The name of the annotation to rename.
        new_name: str
            The new name of the annotation.
        """
        self._annotations.rename(columns={old_name: new_name}, inplace=True)
        if old_name in self.partitions:
            self.partitions.add(new_name)
            self.partitions.remove(old_name)

    def remove_annotation(self, annot_name: str) -> None:
        """Removes a set of annotations from the dataset. This does not
        remove any instances.

        Parameters
        ----------
        annot_name: str
            Annotation name.
        """
        del self._annotations[annot_name]
        # Use difference_update in case annot_name is not already a partition
        self.partitions.difference_update({annot_name})

    def get_annotations(self, annot_name: str) -> np.ndarray:
        """Get a list of annotations, one for each instance currently in
        the dataset.

        Parameters
        ----------
        annot_name: str
            Annotation name.

        Returns
        -------
        A pd.Series of values, one for each instance in the datset, in
        the same order they appear in names and x.
        """
        return self.annotations.loc[self.names, annot_name].to_numpy()

    def get_audio_paths(self) -> np.ndarray:
        return self.get_annotations(_AUDIO_PATH_KEY)

    def get_ratings(self, name: str, rating_set: str = "ratings") -> pd.Series:
        """Get per-annotator ratings of a specified column, for this
        dataset.

        Parameters
        ----------
        name: str
            The name of the rating column to get.
        rating_set: str
            The name of the rating set to get.

        Returns
        -------
        pd.Series
            Pandas Series with (name, rater) multiindex.
        """
        return self.ratings[rating_set].loc[self.names, name]

    def annotation_type(self, annot_name: str) -> type:
        return self.annotations[annot_name].dtype

    def get_group_indices(self, annot_name: str) -> np.ndarray:
        """Gets the group indices (i.e. indices into the groups array)
        for a given partition.

        Parameters
        ----------
        annot_name: str
            The partition name.

        Returns
        -------
        A NumPy array of group indices for each instance in the dataset.
        """
        # _, idx = np.unique(self.get_annotations(annot_name), return_inverse=True)
        idx = self.annotations.loc[self.names, annot_name].cat.codes.to_numpy(np.int64)
        return idx

    def get_group_counts(self, annot_name: str) -> np.ndarray:
        """Get group counts for a partition.

        Parameters
        ----------
        annot_name: str
            The partition name.

        Returns
        -------
        A NumPy array of counts for the corresponding group in this
        partition.
        """
        counts = (
            self.annotations.loc[self.names, annot_name]
            .value_counts(sort=False)
            .to_numpy()
        )
        return counts

    def get_group_names(self, annot_name: str) -> list[str]:
        """Get the names of groups in a partition.

        Parameters
        ----------
        annot_name: str
            Annotation name.
        """
        return list(self.annotations.loc[self.names, annot_name].cat.categories)

    def remove_instances(
        self, *, drop: Collection[str] = None, keep: Collection[str] = None
    ):
        """Remove instances from dataset. Recalculate annotations,
        partitions, etc.

        Parameters
        ----------
        drop: collection of str
            Instances to drop.
        keep: collection of str
            Instances to keep. Exactly one of drop and keep should be
            given.
        """
        drop = set([] if drop is None else drop)
        keep = set([] if keep is None else keep)

        if (len(drop) == 0) == (len(keep) == 0):
            raise ValueError("Exactly one of drop and keep should be given.")

        if len(keep) == 0:
            keep = self.names.difference(drop, sort=False)

        # Need idx for self.x, instead of ordered_intersect()
        idx = self.get_idx_for_names(keep)
        self._names = self.names[idx]
        if len(self._x) > 0:  # To avoid NumPy DeprecationWarning
            self._x = self.x[idx]
        self._reset_categorical()
        self._verify_annotations(msg="Incomplete annotations after removing instances.")

    def clone(self):
        return self.copy()

    def copy(self):
        new_dataset = type(self).__new__(type(self))
        new_dataset._copy_from_dataset(self)
        return new_dataset

    def normalise(
        self, partition: str, normaliser: TransformerMixin = StandardScaler()
    ):
        """Transforms the data matrix of this dataset in-place using
        some (offline) normalisation method.

        Parameters
        ----------
        normaliser:
            The transform to apply. Must implement fit_transform().
        partition: str
            The partition to apply in a per-group fashion. If "all",
            then perform global normalisation on all the data.
        """

        if partition == "all":
            groups = np.zeros(len(self.x), dtype=int)
        elif partition == "instance":
            groups = np.arange(len(self.x))
        else:
            groups = self.get_group_indices(partition)
        group_transform(self.x, groups, normaliser, inplace=True)

    def pad_arrays(self, pad: int = 32):
        """Pads each array to the nearest multiple of `pad` greater than
        the array size. Assumes axis 0 of x is time.
        """
        logger.info(f"Padding array lengths to nearest multiple of {pad}.")
        self._x = pad_arrays(self.x, pad=pad)

    def clip_arrays(self, length: int):
        """Clips each array to the specified maximum length."""
        logger.info(f"Clipping arrays to max length {length}.")
        self._x = clip_arrays(self.x, length=length, copy=False)

    def frame_arrays(
        self,
        frame_size: int,
        frame_shift: int,
        max_frames: Optional[int] = None,
    ):
        """Create a sequence of frames from the raw signal."""
        logger.info(f"Framing arrays with size {frame_size} and shift {frame_shift}.")
        self._x = frame_arrays(
            self._x,
            frame_size=frame_size,
            frame_shift=frame_shift,
            max_frames=max_frames,
        )

    def transpose_time(self):
        """Transpose the time and feature axis of each instance."""
        logger.info("Transposing time and feature axis of data.")
        self._x = transpose_time(self._x)

    @property
    def corpus(self) -> str:
        """The corpus this dataset represents."""
        return self._corpus

    @property
    def description(self) -> str:
        """The descriptive name of this corpus"""
        return self._description

    @property
    def n_instances(self) -> int:
        """Number of instances in this dataset."""
        return len(self.names)

    @property
    def feature_names(self) -> list[str]:
        """list of feature names."""
        return self._feature_names

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.feature_names)

    @property
    def speaker_names(self) -> list[str]:
        """list of unique speakers in this dataset."""
        return self.get_group_names("speaker")

    @property
    def speaker_counts(self) -> np.ndarray:
        """Number of instances for each speaker."""
        return self.get_group_counts("speaker")

    @property
    def speaker_indices(self) -> np.ndarray:
        """Indices into speakers array of corresponding speaker for each
        instance.
        """
        return self.get_group_indices("speaker")

    @property
    def speakers(self) -> np.ndarray:
        return self.get_annotations("speaker")

    @property
    def n_speakers(self) -> int:
        return len(self.speaker_names)

    @property
    def partitions(self) -> Set[str]:
        """Partitions in this dataset."""
        return self._partitions

    @property
    def annotations(self) -> pd.DataFrame:
        """Full annotation matrix for dataset."""
        return self._annotations.loc[self.names]

    @property
    def ratings(self) -> dict[str, pd.DataFrame]:
        """Full ratings for dataset."""
        return self._ratings

    @property
    def names(self) -> pd.Index:
        """list of instance names."""
        return self._names

    @property
    def subset(self) -> str:
        """Name of clip subset used."""
        return self._subset

    @property
    def subsets(self) -> dict[str, list[str]]:
        """dict from subset name to set of clip names."""
        return self._subsets

    @property
    def x(self) -> np.ndarray:
        """The data matrix."""
        return self._x

    def __len__(self) -> int:
        return self.n_instances

    def update_labels(
        self, labels: Union[PathOrStr, Mapping[str, str], Sequence[str], pd.Series]
    ):
        self.update_annotation(self.label_annot, labels, dtype=str)

    def map_classes(self, mapping: Mapping[str, str]):
        """Modifies classses based on the mapping in map. Keys not
        corresponding to classes are ignored. The new classes will be
        sorted lexicographically.
        """
        self.map_groups(self.label_annot, mapping)

    def remove_classes(
        self, *, drop: Collection[str] = None, keep: Collection[str] = None
    ):
        """Remove instances with labels not in `keep`."""
        self.remove_groups(self.label_annot, drop=drop, keep=keep)

    @property
    def classes(self) -> list[str]:
        """list of unique class labels."""
        return self.get_group_names(self.label_annot)

    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(self.classes)

    @property
    def class_counts(self) -> np.ndarray:
        """Number of instances for each class."""
        return self.get_group_counts(self.label_annot)

    @property
    def labels(self) -> np.ndarray:
        """list of labels for instances."""
        return self.get_annotations(self.label_annot)

    @property
    def label_annot(self) -> str:
        """The annotation used as target label."""
        return self._label_annot

    @label_annot.setter
    def label_annot(self, val: str) -> None:
        self._label_annot = val

    @property
    def y(self) -> np.ndarray:
        """The class label array; one label per instance."""
        if self.label_annot in self.partitions:
            return self.get_group_indices(self.label_annot)
        else:
            return self.get_annotations(self.label_annot)

    def __str__(self):
        s = f"Corpus: {self.corpus}\n"
        s += f"Description: {self.description}\n"
        for part in sorted(self.partitions):
            if part == _AUDIO_PATH_KEY:
                continue
            group_names = self.get_group_names(part)
            s += f"partition '{part}' ({len(group_names)} groups)"
            n_invalid = self.annotations.loc[self.names, part].isna().sum()
            if n_invalid > 0:
                n_valid = len(self) - n_invalid
                s += f" (incomplete [{n_valid} valid])"
            s += "\n"
            if len(group_names) <= 20:
                s += f"\t{dict(zip(group_names, self.get_group_counts(part)))}\n"
        for annot in sorted(self.annotations):
            if annot in self.partitions:
                continue
            s += f"annotation '{annot}'\n"
            annotations = self.get_annotations(annot)
            s += f"\tmin: {annotations.min():.3f}, max: {annotations.max():.3f}, "
            s += f"mean: {annotations.mean():.3f}\n"
        s += f"{self.n_instances} instances\n"
        s += "Subsets:\n" if self.subsets else ""
        for subset, names in self.subsets.items():
            s += "\t*" if subset == self.subset else "\t "
            s += f"{subset}: {len(names)} instances\n"
        s += f"using {self._features} ({self.n_features} features)\n"
        if self.x.dtype == object or len(self.x.shape) == 3:
            lengths = [len(x) for x in self.x]
            s += "Sequences:\n"
            s += f"min length: {np.min(lengths)}\n"
            s += f"mean length: {np.mean(lengths)}\n"
            s += f"max length: {np.max(lengths)}\n"
        return s


class CombinedDataset(Dataset):
    """A dataset that joins one or more individual datasets together and
    merges annotations.
    """

    def __init__(self, *datasets: Dataset):
        if len(datasets) == 0:
            raise ValueError("No datasets provided.")
        if len(datasets) == 1:
            self._copy_from_dataset(datasets[0])
            return

        self._init()

        combined_str = ", ".join(d.corpus for d in datasets)
        logger.info(f"Combining {combined_str}")
        self._corpus = "combined"
        self._description = f"Combined datasets ({combined_str})"
        self._names = pd.Index([f"{d.corpus}_{n}" for d in datasets for n in d.names])
        self._feature_names = datasets[0].feature_names
        self._features = datasets[0]._features
        self._x = np.concatenate([d.x for d in datasets])
        self.update_annotation(
            "corpus", [d.corpus for d in datasets for _ in d.names], dtype="category"
        )

        all_speakers = {}
        for d in datasets:
            if "speaker" not in d.annotations.columns:
                all_speakers.update(
                    {f"{d.corpus}_{k}": f"{d.corpus}_unknown" for k in d.names}
                )
            else:
                speakers = d.annotations["speaker"]
                all_speakers.update(
                    {f"{d.corpus}_{k}": f"{d.corpus}_{v}" for k, v in speakers.items()}
                )
        self.update_annotation("speaker", all_speakers)

        # TODO: handle case when only some datasets have a given annotation?
        all_annotations = set(chain(*(d.annotations.columns for d in datasets)))
        all_annotations -= {"speaker", "corpus"}  # assumed to be per-dataset
        common_annotations = {
            x
            for x in all_annotations
            if all(x in d.annotations.columns for d in datasets)
        }
        logger.info(f"All annotations: {all_annotations}")
        logger.info(f"Common annotations: {common_annotations}")
        for annot_name in common_annotations:
            combined_annot = {}
            for dataset in datasets:
                annotations = dataset.annotations[annot_name]
                combined_annot.update(
                    {f"{dataset.corpus}_{k}": v for k, v in annotations.items()}
                )
            self.update_annotation(annot_name, combined_annot)

        # Non-common categorical annotations
        for annot_name in all_annotations - common_annotations:
            if not any(annot_name in d.partitions for d in datasets):
                continue
            combined_annot = {}
            for d in datasets:
                if annot_name not in d.annotations:
                    combined_annot.update({f"{d.corpus}_{k}": None for k in d.names})
                else:
                    annotations = d.annotations[annot_name]
                    combined_annot.update(
                        {f"{d.corpus}_{k}": v for k, v in annotations.items()}
                    )
            self.update_annotation(annot_name, combined_annot)

        if any(d.label_annot == "" for d in datasets):
            return

        if any(d.label_annot != datasets[0].label_annot for d in datasets):
            # Try and unify label annotations with different names
            warnings.warn(
                "label annotation differs between some datasets, attempting to unify."
            )
            combined_labels = {}
            for d in datasets:
                for name in d.names:
                    label = d.annotations[d.label_annot][name]
                    combined_labels[f"{d.corpus}_{name}"] = label
            self.update_annotation("_combined_labels", combined_labels)
            self.label_annot = "_combined_labels"
        else:
            self.label_annot = next(d.label_annot for d in datasets)

    @property
    def corpus_names(self) -> list[str]:
        """list of corpora in this CombinedDataset."""
        return self.get_group_names("corpus")

    @property
    def corpus_indices(self) -> np.ndarray:
        """Indices into corpora list of corresponding corpus for each
        instance.
        """
        return self.get_group_indices("corpus")


def load_multiple(
    corpus_files: Iterable[PathOrStr],
    features: Optional[str] = None,
    subsets: Union[str, Mapping[str, str]] = "default",
    label: Union[str, Mapping[str, str]] = "label",
    **read_kwargs,
) -> CombinedDataset:
    """Load one or more datasets with the given features.

    Parameters
    ----------
    corpus_files: iterable
        The corpus description YAML files to load.
    features: str
        A common set of features to load. This will be found in the
        features directory corresponding to each corpus.
    subsets: str or dict
        A subset name common to all datasets (e.g. "all", "default") or
        a mapping from dataset name to subset name.
    **read_kwargs:
        Other args to pass to feature loading.
    """
    corpus_files = list(corpus_files)
    if len(corpus_files) == 0:
        raise RuntimeError("No corpus metadata files given.")
    datasets = []
    for file in corpus_files:
        logger.info(f"Loading {file}")
        dataset = Dataset(file, subset="all")
        if isinstance(label, str):
            setattr(dataset, "label_annot", label)
        else:
            setattr(dataset, "label_annot", label.get(dataset.corpus, "label"))
        if isinstance(subsets, str):
            dataset.use_subset(subsets)
        else:
            dataset.use_subset(subsets.get(dataset.corpus, "default"))
        if features is not None:
            dataset.update_features(features, **read_kwargs)
        datasets.append(dataset)
    return CombinedDataset(*datasets)


def load_datasets_config(config: Union[PathOrStr, DataLoadConfig]) -> Dataset:
    """Load one or more datasets from a DataLoadConfig.

    Parameters
    ----------
    config: DataLoadConfig
        The data loading config.

    Returns
    -------
    Dataset
        A dataset that combines one or more datasets according to the
        given config.
    """
    if isinstance(config, (str, os.PathLike)):
        config = DataLoadConfig.from_file(config)
    datasets: list[Dataset] = []
    for corpus, dataset_conf in config.datasets.items():
        logger.info(f"Loading {dataset_conf.path}")
        dataset = Dataset(dataset_conf.path, subset=dataset_conf.subset)
        dataset._corpus = corpus
        for part, mapping in dataset_conf.map_groups.items():
            dataset.map_groups(part, mapping.map)
        for part, remove in dataset_conf.remove_groups.items():
            dataset.remove_groups(part, drop=remove.drop, keep=remove.keep)
        for a1, a2 in dataset_conf.rename_annotations.items():
            # TODO: check for collisions
            dataset.rename_annotation(a1, a2)
        if dataset_conf.features:
            dataset.update_features(dataset_conf.features, **dataset_conf.read_kwargs)
        elif config.features:
            dataset.update_features(config.features, **dataset_conf.read_kwargs)
        if dataset_conf.clip_seq:
            dataset.clip_arrays(dataset_conf.clip_seq)
        if dataset_conf.pad_seq:
            dataset.pad_arrays(dataset_conf.pad_seq)
        datasets.append(dataset)
    if len(config.datasets) > 1:
        dataset = CombinedDataset(*datasets)
        del datasets
    else:
        dataset = datasets[0]
    for part, mapping in config.map_groups.items():
        dataset.map_groups(part, mapping.map)
    for part, remove in config.remove_groups.items():
        dataset.remove_groups(part, drop=remove.drop, keep=remove.keep)
    if config.clip_seq:
        dataset.clip_arrays(config.clip_seq)
    if config.pad_seq:
        dataset.pad_arrays(config.pad_seq)
    if config.label:
        dataset.label_annot = config.label
    return dataset
