import copy
import logging
import os
import warnings
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)

import numpy as np
import yaml
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing_extensions import Literal

from ertk.config import get_arg_mapping
from ertk.dataset.annotation import read_annotations
from ertk.dataset.features import find_features_file, read_features
from ertk.dataset.utils import get_audio_paths
from ertk.transform import group_transform
from ertk.utils import (
    PathOrStr,
    clip_arrays,
    frame_arrays,
    ordered_intersect,
    pad_arrays,
    transpose_time,
)


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

    _annotations: Dict[str, Dict[str, Any]]
    _corpus: str = ""
    _feature_names: List[str]
    _names: List[str]
    _partitions: Set[str]
    _subset: str = ""
    _subsets: Dict[str, List[str]]
    _x: np.ndarray

    # "Private" vars
    _default_subset: str = ""
    _features: str = ""
    _features_path: Path
    _features_dir: Path
    _subset_paths: Dict[str, Path]

    def __init__(
        self,
        corpus_info: PathOrStr,
        features: Optional[PathOrStr] = None,
        subset: str = "default",
    ):
        self._init()
        self.init_corpus_info(corpus_info)
        self.use_subset(subset)
        if features is not None:
            self.update_features(features)

    def _init(self) -> None:
        self._annotations = {}
        self._feature_names = []
        self._names = []
        self._partitions = set()
        self._subsets = {}
        self._subset_paths = {}
        self._x = np.empty((0, 0), dtype=np.float32)

    @classmethod
    def _copy(cls, inst: "Dataset"):
        new_dataset = Dataset.__new__(cls)
        new_dataset._copy_from_dataset(inst)
        return new_dataset

    def _copy_from_dataset(self, other: "Dataset") -> None:
        self._annotations = copy.deepcopy(other._annotations)
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
        path = Path(path)
        with open(path) as fid:
            doc = yaml.safe_load(fid)
        if not isinstance(doc, dict):
            raise RuntimeError("Corpus info invalid.")
        self._corpus = next(iter(doc))
        corpus_info = doc[self.corpus]

        self._features_dir = path.parent / corpus_info["features_dir"]
        self._default_subset = corpus_info["subsets"]["default"]
        for subset, subset_info in corpus_info["subsets"].items():
            if subset == "default":
                continue
            clips_file = path.parent / f"{subset_info['clips']}.txt"
            self._subset_paths[subset] = clips_file
            self.subsets[subset] = [x.stem for x in get_audio_paths(clips_file)]

        if corpus_info["partitions"]:
            for partition in corpus_info["partitions"]:
                self.update_annotation(
                    partition, path.parent / f"{partition}.csv", dtype=str
                )
        if corpus_info["annotations"]:
            for annotation in corpus_info["annotations"]:
                self.update_annotation(annotation, path.parent / f"{annotation}.csv")

    def _verify_annotations(self, annot_name: Optional[str] = None):
        """Make sure there is a value for each instance for each annotation."""

        def verify_annotation(annot_name: str):
            missing = set(self.names) - self.annotations[annot_name].keys()
            if len(missing) > 0:
                raise RuntimeError(
                    f"Incomplete annotation {annot_name}. These names are missing:"
                    f"{missing}"
                )

        to_verify = self.annotations.keys() if annot_name is None else {annot_name}
        for annot_name in to_verify:
            verify_annotation(annot_name)

    def update_features(self, features: PathOrStr) -> None:
        """Update the features matrix and feature names for this dataset.

        Parameters
        ----------
        features: os.PathLike or str
            Path to a set of features or unique name of features in
            corpus features dir.
        """
        if features == "raw_audio":
            self._features = "raw_audio"
            features = self._subset_paths[self.subset or "all"]
        else:
            features = Path(features)
            if len(features.suffix) == 0:
                features = find_features_file(self._features_dir.glob(f"{features}.*"))
            self._features = features.stem
        self._features_path = features
        data = read_features(features)
        self._feature_names = data.feature_names
        if len(self.names) > 0:
            names = set(self.names)
            if len(names - set(data.names)) > 0:
                raise ValueError(
                    f"Features at {features} don't contain all instances for subset "
                    f"{self.subset}."
                )
            self._x = data.features[[i for i, n in enumerate(data.names) if n in names]]
            self._names = ordered_intersect(data.names, names)
        else:
            self._x = data.features
            self._names = data.names

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
        self._names = self.subsets[subset]
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
        names = set(names)
        return np.array([i for i, x in enumerate(self.names) if x in names])

    @overload
    def get_idx_for_split(
        self,
        split: Union[str, Dict[str, Union[str, Collection[str]]]],
        return_complement: Literal[False],
    ) -> np.ndarray:
        ...

    @overload
    def get_idx_for_split(
        self,
        split: Union[str, Dict[str, Union[str, Collection[str]]]],
        return_complement: Literal[True],
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def get_idx_for_split(
        self,
        split: Union[str, Dict[str, Union[str, Collection[str]]]],
        return_complement: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
        if isinstance(split, str):
            if ":" not in split and not Path(split).exists():  # is a subset
                return self.get_idx_for_names(self.subsets[split])
            mapping = get_arg_mapping(split)
        else:
            mapping = split
        indices: Set[int] = set()
        for part_name in mapping:
            group_names = self.get_group_names(part_name)
            group_indices = self.get_group_indices(part_name)
            sel = mapping[part_name]
            sel = [sel] if isinstance(sel, str) else sel
            sel_idx = [group_names.index(x) for x in sel]
            indices.update(np.flatnonzero(np.isin(group_indices, sel_idx)))
        comp_indices = set(range(len(self))) - indices
        if return_complement:
            return np.array(sorted(indices)), np.array(sorted(comp_indices))
        else:
            return np.array(sorted(indices))

    def update_annotation(
        self,
        annot_name: str,
        annotations: Union[PathOrStr, Mapping[str, Any], Sequence[Any]],
        dtype: Optional[Type] = None,
    ) -> None:
        """Add or update an annotation.

        Parameters
        ----------
        annot_name: str
            The name of the annotation.
        annotations: PathLike, str, mapping or sequence
            Annotations to add, similar to update_partition(). If
            PathLike or str, annotations are read from a CSV. If a dict,
            should be of the form {instance: annotation}. If a list,
            should have an annotation for each instance.
        dtype: type, optional
            The type of annotations for reading from CSV file.
        """
        if isinstance(annotations, (os.PathLike, str)):
            annot_dict = read_annotations(annotations, dtype=dtype)
        elif isinstance(annotations, Mapping):
            annot_dict = dict(annotations)
        else:
            annot_dict = {
                k: v for k, v in zip(self.names, annotations) if v is not None
            }

        self._annotations[annot_name] = annot_dict
        if (
            dtype is not None
            and issubclass(dtype, str)
            or isinstance(next(iter(annot_dict.values())), str)
        ):
            self.partitions.add(annot_name)
        self._verify_annotations(annot_name)

    def map_groups(self, part_name: str, mapping: Mapping[str, str]) -> None:
        """Map group names in a partition.

        Parameters
        ----------
        part_name: str
            Name of partition.
        mapping: dict
            Group name mapping.
        """
        groups = self.get_annotations(part_name)
        self.update_annotation(
            part_name, [mapping.get(x, x) for x in groups], dtype=str
        )

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
        groups: collection of str
            The groups to remove in given partition.
        """
        drop = set([] if drop is None else drop)
        keep = set([] if keep is None else keep)

        if (len(drop) == 0) == (len(keep) == 0):
            raise ValueError("Exactly one of drop and keep should be non-empty.")

        if len(keep) == 0:
            keep = set(self.get_group_names(part_name)) - drop

        annotation = self.annotations[part_name]
        self.remove_instances(keep=[x for x in self.names if annotation[x] in keep])

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
        self.annotations[new_name] = self.annotations[old_name]
        del self.annotations[old_name]
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
        del self.annotations[annot_name]
        # Use difference_update in case annot_name is not already a partition
        self.partitions.difference_update({annot_name})

    def get_annotations(self, annot_name: str) -> List[Any]:
        """Get a list of annotations, one for each instance currently in
        the dataset.

        Parameters
        ----------
        annot_name: str
            Annotation name.

        Returns
        -------
        A list of values, one for each instance in the datset, in the
        same order they appear in names and x.
        """
        return [self.annotations[annot_name][x] for x in self.names]

    def annotation_type(self, annot_name: str) -> Type:
        return type(next(iter(self.annotations[annot_name].values())))

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
        _, idx = np.unique(self.get_annotations(annot_name), return_inverse=True)
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
        _, counts = np.unique(self.get_annotations(annot_name), return_counts=True)
        return counts

    def get_group_names(self, annot_name: str) -> List[str]:
        """Get the names of groups in a partition.

        Parameters
        ----------
        annot_name: str
            Annotation name.
        """
        return list(np.unique(self.get_annotations(annot_name)))

    def update_speakers(
        self, speakers: Union[PathOrStr, Mapping[str, str], Sequence[str]]
    ):
        """Update the speakers for this dataset.

        Parameters
        ----------
        speakers: Path, str, dict or list
            If Path or str, read speaker info from CSV at given path. If
            dict, use speaker dict directly. If list, each speakers[i]
            is the corresponding speaker for instance i.
        """
        self.update_annotation("speaker", speakers, dtype=str)

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
            keep = set(self.names) - drop

        idx = self.get_idx_for_names(keep)
        self._names = [self.names[i] for i in idx]
        if len(self._x) > 0:  # To avoid NumPy DeprecationWarning
            self._x = self.x[idx]
        try:
            self._verify_annotations()
        except RuntimeError as e:
            raise RuntimeError(
                "Incomplete annotations after removing instances."
            ) from e

    def clone(self):
        return self.copy()

    def copy(self):
        return self._copy(self)

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
        logging.info(f"Padding array lengths to nearest multiple of {pad}.")
        self._x = pad_arrays(self.x, pad=pad)

    def clip_arrays(self, length: int):
        """Clips each array to the specified maximum length."""
        logging.info(f"Clipping arrays to max length {length}.")
        self._x = clip_arrays(self.x, length=length, copy=False)

    def frame_arrays(
        self,
        frame_size: int,
        frame_shift: int,
        max_frames: Optional[int] = None,
    ):
        """Create a sequence of frames from the raw signal."""
        logging.info(f"Framing arrays with size {frame_size} and shift {frame_shift}.")
        self._x = frame_arrays(
            self._x,
            frame_size=frame_size,
            frame_shift=frame_shift,
            max_frames=max_frames,
        )

    def transpose_time(self):
        """Transpose the time and feature axis of each instance."""
        logging.info("Transposing time and feature axis of data.")
        self._x = transpose_time(self._x)

    @property
    def corpus(self) -> str:
        """The corpus this dataset represents."""
        return self._corpus

    @property
    def n_instances(self) -> int:
        """Number of instances in this dataset."""
        return len(self.names)

    @property
    def feature_names(self) -> List[str]:
        """List of feature names."""
        return self._feature_names

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.feature_names)

    @property
    def speaker_names(self) -> List[str]:
        """List of unique speakers in this dataset."""
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
    def speakers(self) -> List[str]:
        return self.get_annotations("speaker")

    @property
    def n_speakers(self) -> int:
        return len(self.speaker_names)

    @property
    def partitions(self) -> Set[str]:
        """Partitions in this dataset."""
        return self._partitions

    @property
    def annotations(self) -> Dict[str, Dict[str, Any]]:
        """Each annotation is a mapping from instance name to value."""
        return self._annotations

    @property
    def names(self) -> List[str]:
        """List of instance names."""
        return self._names

    @property
    def subset(self) -> str:
        """Name of clip subset used."""
        return self._subset

    @property
    def subsets(self) -> Dict[str, List[str]]:
        """Dict from subset name to set of clip names."""
        return self._subsets

    @property
    def x(self) -> np.ndarray:
        """The data matrix."""
        return self._x

    def __len__(self) -> int:
        return self.n_instances

    def __str__(self):
        s = f"Corpus: {self.corpus}\n"
        for part in sorted(self.partitions):
            group_names = self.get_group_names(part)
            s += f"partition '{part}' ({len(group_names)} groups)\n"
            if len(group_names) <= 20:
                s += f"\t{dict(zip(group_names, self.get_group_counts(part)))}\n"
        for annot in sorted(self.annotations):
            if annot in self.partitions:
                continue
            s += f"annotation '{annot}'\n"
            annotations = np.array(self.get_annotations(annot))
            s += f"\tmin: {annotations.min():.3f}, max: {annotations.max():.3f}, "
            s += f"mean: {annotations.mean():.3f}\n"
        s += f"{self.n_instances} instances\n"
        s += f"subset: {self.subset}\n"
        s += f"using {self._features} ({self.n_features} features)\n"
        if self.x.dtype == object or len(self.x.shape) == 3:
            lengths = [len(x) for x in self.x]
            s += "Sequences:\n"
            s += f"min length: {np.min(lengths)}\n"
            s += f"mean length: {np.mean(lengths)}\n"
            s += f"max length: {np.max(lengths)}\n"
        return s


class LabelledDataset(Dataset):
    """Class representing a dataset containing discrete labels for each
    instance.
    """

    _label_annot: str

    def __init__(
        self,
        corpus_info: PathOrStr,
        features: Optional[PathOrStr] = None,
        subset: str = "default",
        label: str = "label",
    ):
        super().__init__(corpus_info, features, subset)
        self._label_annot = label

    def _copy_from_dataset(self, other: "Dataset") -> None:
        super()._copy_from_dataset(other)
        self._label_annot = getattr(other, "_label_annot", "label")

    def update_labels(self, labels: Union[PathOrStr, Mapping[str, str], Sequence[str]]):
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
    def classes(self) -> List[str]:
        """List of unique class labels."""
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
    def labels(self) -> List[str]:
        """List of labels for instances."""
        return self.get_annotations(self.label_annot)

    @property
    def label_annot(self) -> str:
        """The annotation used as target label."""
        return self._label_annot

    @label_annot.setter
    def label_annot(self, val: str) -> None:
        if val not in self.annotations:
            raise ValueError(f"'{val}' not in annotations.")
        self._label_annot = val

    @property
    def y(self) -> np.ndarray:
        """The class label array; one label per instance."""
        if self.label_annot in self.partitions:
            return self.get_group_indices(self.label_annot)
        else:
            return np.ndarray(self.get_annotations(self.label_annot))


class CombinedDataset(LabelledDataset):
    """A dataset that joins one or more individual datasets together and
    merges annotations.
    """

    def __init__(self, *datasets: LabelledDataset):
        if len(datasets) == 0:
            raise ValueError("No datasets provided.")
        if len(datasets) == 1:
            self._copy_from_dataset(datasets[0])
            return

        self._init()

        logging.info("Combining " + ", ".join(d.corpus for d in datasets))
        self._corpus = "combined"
        self._names = [f"{d.corpus}_{n}" for d in datasets for n in d.names]
        self._feature_names = datasets[0].feature_names
        self._features = datasets[0]._features
        self._x = np.concatenate([d.x for d in datasets])

        self.update_annotation(
            "corpus", [d.corpus for d in datasets for _ in d.names], dtype=str
        )

        all_speakers = {}
        for d in datasets:
            if "speaker" not in d.annotations:
                all_speakers.update(
                    {f"{d.corpus}_{k}": f"{d.corpus}_unknown" for k in d.names}
                )
            else:
                speakers = d.annotations["speaker"]
                all_speakers.update(
                    {f"{d.corpus}_{k}": f"{d.corpus}_{v}" for k, v in speakers.items()}
                )
        self.update_speakers(all_speakers)

        # TODO: handle case when only some datasets have a given annotation?
        all_annotations = set(chain(*(d.annotations for d in datasets)))
        all_annotations -= {"speaker"}  # assumed to be per-dataset
        common_annotations = {
            x for x in all_annotations if all(x in d.annotations for d in datasets)
        }
        logging.info(f"All annotations: {all_annotations}")
        logging.info(f"Common annotations: {common_annotations}")
        for annot_name in common_annotations:
            combined_annot = {}
            for dataset in datasets:
                annotations = dataset.annotations[annot_name]
                combined_annot.update(
                    {f"{dataset.corpus}_{k}": v for k, v in annotations.items()}
                )
            self.update_annotation(annot_name, combined_annot)

        # Not common categorical annotations
        for annot_name in all_annotations - common_annotations:
            if not any(annot_name in d.partitions for d in datasets):
                continue
            combined_annot = {}
            for d in datasets:
                if annot_name not in d.annotations:
                    combined_annot.update(
                        {f"{d.corpus}_{k}": "unknown" for k in d.names}
                    )
                else:
                    annotations = d.annotations[annot_name]
                    combined_annot.update(
                        {f"{d.corpus}_{k}": v for k, v in annotations.items()}
                    )
            self.update_annotation(annot_name, combined_annot)

        if any(d.label_annot != datasets[0].label_annot for d in datasets):
            # Try and unify label annotations with different names
            warnings.warn(
                "label annotation differs between some datasets, attempting to unify."
            )
            self._label_annot = "_combined_labels"
            combined_labels = {}
            for d in datasets:
                for name in d.names:
                    label = d.annotations[d.label_annot][name]
                    combined_labels[f"{d.corpus}_{name}"] = label
            self.update_labels(combined_labels)
        else:
            self._label_annot = next(d.label_annot for d in datasets)

    @property
    def corpus_names(self) -> Sequence[str]:
        """List of corpora in this CombinedDataset."""
        return self.get_group_names("corpus")

    @property
    def corpus_indices(self) -> np.ndarray:
        """Indices into corpora list of corresponding corpus for each
        instance.
        """
        return self.get_group_indices("corpus")


def load_multiple(
    corpus_files: Iterable[PathOrStr],
    features: str,
    subsets: Union[str, Mapping[str, str]] = "default",
    label: Union[str, Mapping[str, str]] = "label",
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
    """
    corpus_files = list(corpus_files)
    if len(corpus_files) == 0:
        raise RuntimeError("No corpus metadata files given.")
    datasets = []
    for file in corpus_files:
        logging.info(f"Loading {file}")
        dataset = LabelledDataset(file, subset="all")
        if isinstance(label, str):
            dataset.label_annot = label
        else:
            dataset.label_annot = label.get(dataset.corpus, "label")
        if isinstance(subsets, str):
            dataset.use_subset(subsets)
        else:
            dataset.use_subset(subsets.get(dataset.corpus, "default"))
        dataset.update_features(features)
        datasets.append(dataset)
    return CombinedDataset(*datasets)
