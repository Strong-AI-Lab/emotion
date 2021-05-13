import os
import warnings
from itertools import chain
from typing import (
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

from ..utils import (
    PathOrStr,
    clip_arrays,
    flat_to_inst,
    frame_arrays,
    inst_to_flat,
    pad_arrays,
    transpose_time,
)
from .annotation import read_annotations
from .features import read_features

_AT = TypeVar("_AT", str, int, float)

_AnnotationCatsList = Union[Sequence[str], Sequence[int]]
_AnnotationList = Union[
    Sequence[Optional[str]], Sequence[Optional[int]], Sequence[Optional[float]]
]


class Dataset:
    """Class representing a generic dataset, consisting of a set of
    features and optional partitions and annotations. Has various
    preprocessing methods.

    An annotation is a scalar value for some (or all) instances in the
    dataset. Annotations are allowed to be incomplete, which occurs when
    majority vote labels cannot be assigned to all instances, for
    example.

    A partition is a partition of instances into disjoint groups (e.g.
    speakers). A partition should be complete in that each instance is
    in exactly one group in the partition. Each partition has a
    corresponding annotation with the same name, using the group names
    as categorical annotations.
    """

    _partitions: Dict[str, Dict[str, Set[str]]]
    _annotation_categories: Dict[str, _AnnotationCatsList]
    _annotations: Dict[str, _AnnotationList]
    _corpus = ""
    _names: List[str]
    _feature_names: List[str]
    _x: np.ndarray

    def __init__(self, path: PathOrStr, speaker_path: Optional[PathOrStr] = None):
        data = read_features(path)

        self._corpus = data.corpus
        self._names = data.names
        self._feature_names = data.feature_names
        self._x = data.features

        self._partitions = {}
        self._annotation_categories = {}
        self._annotations = {}

        if speaker_path:
            self.update_speakers(speaker_path)
        else:
            self.update_speakers(["unknown"] * len(self.names))

    def get_annotation_indices(self, annot_name: str) -> np.ndarray:
        """Get annotation indices for categorical annotations (e.g.
        speakers).
        """
        if annot_name not in self.annotation_categories:
            raise ValueError(f"annotation {annot_name} doesn't have categories")
        annotations = self.annotations[annot_name]
        cats = self.annotation_categories[annot_name]
        return np.array([cats.index(x) for x in annotations])

    def get_annotation_counts(self, annot_name: str) -> np.ndarray:
        """Get counts of discrete annotation values."""
        uniq, counts = np.unique(self.annotations[annot_name], return_counts=True)
        count_dict = dict(zip(uniq, counts))
        return np.array([count_dict[x] for x in self.annotation_categories[annot_name]])

    def update_annotation(
        self,
        annot_name: str,
        annotations: Union[PathOrStr, Mapping[str, _AT], Sequence[_AT]],
        dtype: Optional[Type[_AT]] = None,
    ):
        """Update or add an annotation."""
        if isinstance(annotations, (os.PathLike, str)):
            if dtype is None:
                raise TypeError(
                    "dtype must be specified when adding annotations from CSV"
                )
            annot_dict: Mapping[str, _AT] = read_annotations(annotations, dtype=dtype)
        elif isinstance(annotations, Mapping):
            annot_dict = annotations
        else:
            annot_dict = {
                k: v for k, v in zip(self.names, annotations) if v is not None
            }

        if not set(self.names) <= set(annot_dict.keys()):
            warnings.warn(f"Adding incomplete annotation {annot_name}.")
        self._annotations[annot_name] = [annot_dict.get(name) for name in self.names]
        type_ = type(annot_dict[self.names[0]])
        if issubclass(type_, (str, int)):
            self._annotation_categories[annot_name] = sorted(set(annot_dict.values()))

    def get_group_indices(self, part_name: str) -> np.ndarray:
        """Gets the group indices (i.e. indices into the groups array)
        for a given partition.

        Args:
        -----
        part_name: str
            The partition name.

        Returns:
        --------
        A NumPy array of group indices for each instance in the dataset.
        """
        return self.get_annotation_indices(part_name)

    def get_group_counts(self, part_name: str) -> np.ndarray:
        """Get group counts for a partition.

        Args:
        -----
        part_name: str
            The partition name.

        Returns:
        --------
        A NumPy array of counts for the corresponding group in this
        partition.
        """
        return np.array([len(g) for g in self.partitions[part_name].values()])

    def update_partition(
        self, part_name: str, groups: Union[PathOrStr, Mapping[str, str], Sequence[str]]
    ):
        """Add or update partition of instances in this dataset.

        Args:
        -----
        part_name: str
            The partition name.
        groups: PathLike or str or dict or list
            The groups for this partition. If PathLike, groups are read
            from a CSV. If a dict, should be of the form {instance:
            group}. If a list, should have a group for each instance.

            The groups should be complete (i.e. the union of the groups
            should contain all instances) and mutually exclusive (no two
            groups should contain the same instance).
        """
        if isinstance(groups, (os.PathLike, str)):
            grp_dict = read_annotations(groups, dtype=str)
            names, grp = zip(*grp_dict.items())
        elif isinstance(groups, Mapping):
            names, grp = zip(*groups.items())
        else:
            names = tuple(self.names)
            grp = tuple(groups)

        if not set(self.names) <= set(names):
            raise ValueError("Partition must cover all instances in the dataset.")

        unique, idx = np.unique(grp, return_inverse=True)
        self._partitions[part_name] = {}
        for i, g_name in enumerate(unique):
            name_idx = [j for j in range(len(idx)) if idx[j] == i]
            names_group = {names[j] for j in name_idx}.intersection(self.names)
            self._partitions[part_name][g_name] = names_group
        self.update_annotation(part_name, groups, str)

    def update_speakers(self, speakers: Union[PathOrStr, Dict[str, str], List[str]]):
        """Update the speakers for this dataset.

        Args:
        -----
        speakers: Path, str, dict or list
            If Path or str, read speaker info from CSV at given path. If
            dict, use speaker dict directly. If list, each speakers[i]
            is the corresponding speaker for instance i.
        """
        self.update_partition("speakers", speakers)
        self._update_speakers()

    def _update_speakers(self):
        self._speaker_indices = self.get_group_indices("speakers")
        if any(x == 0 for x in self.speaker_counts):
            warnings.warn("Some speakers have no corresponding instances.")

    def remove_instances(
        self, *, drop: Collection[str] = [], keep: Collection[str] = []
    ):
        """Remove instances from dataset. Recalculate annotations,
        partitions, etc.

        Args:
        -----
        drop: collection of str
            Instances to drop.
        keep: collection of str
            Instances to keep. Exactly one of drop and keep should be
            non-empty.
        """
        if bool(drop) == bool(keep):
            raise ValueError("Exactly one of drop and keep should be non-empty.")

        if len(drop) > 0:
            keep = set(self.names) - set(drop)

        idx = [i for i, x in enumerate(self.names) if x in keep]
        self._names = [self.names[i] for i in idx]
        self._x = self.x[idx]

        for annot_name in self.annotations:
            new_annotations = [self.annotations[annot_name][i] for i in idx]
            if annot_name in self.partitions:
                # This will also update annotations
                self.update_partition(annot_name, new_annotations)
            else:
                self.update_annotation(annot_name, new_annotations)
        self._update_speakers()

    def normalise(
        self, normaliser: TransformerMixin = StandardScaler(), scheme: str = "speaker"
    ):
        """Transforms the X data matrix of this dataset using some
        (global) normalisation method. I think in theory this should be
        idempotent.
        """

        if scheme == "all":
            flat, slices = inst_to_flat(self.x)
            flat = normaliser.fit_transform(flat)
            self._x = flat_to_inst(flat, slices)
        elif scheme == "speaker":
            for sp in range(len(self.speaker_names)):
                idx = np.nonzero(self.speaker_indices == sp)[0]
                flat, slices = inst_to_flat(self.x[idx])
                flat = normaliser.fit_transform(flat)
                self.x[idx] = flat_to_inst(flat, slices)

    def pad_arrays(self, pad: int = 32):
        """Pads each array to the nearest multiple of `pad` greater than
        the array size. Assumes axis 0 of x is time.
        """
        print(f"Padding array lengths to nearest multiple of {pad}.")
        pad_arrays(self.x, pad=pad)

    def clip_arrays(self, length: int):
        """Clips each array to the specified maximum length."""
        print(f"Clipping arrays to max length {length}.")
        clip_arrays(self.x, length=length)

    def frame_arrays(
        self,
        frame_size: int = 640,
        frame_shift: int = 160,
        num_frames: Optional[int] = None,
    ):
        """Create a sequence of frames from the raw signal."""
        print(f"Framing arrays with size {frame_size} and shift {frame_shift}.")
        self._x = frame_arrays(
            self._x,
            frame_size=frame_size,
            frame_shift=frame_shift,
            num_frames=num_frames,
        )

    def transpose_time(self):
        """Transpose the time and feature axis of each instance."""
        print("Transposing time and feature axis of data.")
        self._x = transpose_time(self._x)

    @property
    def corpus(self) -> str:
        """The corpus this LabelledDataset represents."""
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
    def speaker_names(self) -> Sequence[str]:
        """List of unique speakers in this dataset."""
        return self.annotation_categories["speakers"]

    @property
    def speaker_counts(self) -> np.ndarray:
        """Number of instances for each speaker."""
        return self.get_group_counts("speakers")

    @property
    def speaker_indices(self) -> np.ndarray:
        """Indices into speakers array of corresponding speaker for each
        instance.
        """
        return self._speaker_indices

    @property
    def speakers(self) -> Sequence[str]:
        return self.annotations["speakers"]

    @property
    def n_speakers(self) -> int:
        return len(self.speaker_names)

    @property
    def partitions(self) -> Dict[str, Dict[str, Set[str]]]:
        """Mapping of instance partitions."""
        return self._partitions

    @property
    def part_names(self) -> List[str]:
        return list(self.partitions.keys())

    @property
    def annotations(self) -> Dict[str, _AnnotationList]:
        """Mapping of annotations."""
        return self._annotations

    @property
    def annotation_categories(self) -> Dict[str, _AnnotationCatsList]:
        """List of categories for categorical annotations."""
        return self._annotation_categories

    @property
    def names(self) -> List[str]:
        """List of instance names."""
        return self._names

    @property
    def x(self) -> np.ndarray:
        """The data matrix."""
        return self._x

    def __len__(self) -> int:
        return self.n_instances

    def __str__(self):
        s = f"\nCorpus: {self.corpus}\n"
        s += f"{self.n_instances} instances\n"
        s += f"{len(self.feature_names)} features\n"
        s += f"{len(self.speaker_names)} speakers:\n"
        s += f"\t{dict(zip(self.speaker_names, self.speaker_counts))}\n"
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

    _y: np.ndarray

    def __init__(
        self,
        path: PathOrStr,
        label_path: Optional[PathOrStr] = None,
        speaker_path: Optional[PathOrStr] = None,
    ):
        super().__init__(path, speaker_path=speaker_path)
        if label_path:
            self.update_labels(label_path)
        else:
            self.update_labels(["default"] * len(self.names))
        self._y = self.get_annotation_indices("labels")

    def update_labels(self, labels: Union[PathOrStr, Mapping[str, str], Sequence[str]]):
        # TODO: Allow partial labels, etc.
        self.update_partition("labels", labels)

    def remove_instances(
        self, *, drop: Collection[str] = [], keep: Collection[str] = []
    ):
        super().remove_instances(drop=drop, keep=keep)
        self._y = self.get_annotation_indices("labels")

    def map_classes(self, mapping: Mapping[str, str]):
        """Modifies classses based on the mapping in map. Keys not
        corresponding to classes are ignored. The new classes will be
        sorted lexicographically.
        """
        new_labels = [mapping.get(x, x) for x in self.labels]
        self.update_labels(new_labels)
        self._y = self.get_annotation_indices("labels")

    def remove_classes(self, *, drop: Collection[str] = [], keep: Collection[str] = []):
        """Remove instances with labels not in `keep`."""
        if bool(drop) == bool(keep):
            raise ValueError("Exactly one of drop and keep must be non-empty.")

        if len(drop) > 0:
            keep = set(self.classes) - set(drop)

        keep = set(keep)
        keep_inst = [n for n, l in zip(self.names, self.labels) if l in keep]
        self.remove_instances(keep=keep_inst)

    @property
    def classes(self) -> Sequence[str]:
        """A list of emotion class labels."""
        return self.annotation_categories["labels"]

    @property
    def n_classes(self) -> int:
        """Total number of emotion classes."""
        return len(self.classes)

    @property
    def class_counts(self) -> np.ndarray:
        """Number of instances for each class."""
        return self.get_annotation_counts("labels")

    @property
    def labels(self) -> Sequence[str]:
        """Mapping from instance to label."""
        return self.annotations["labels"]

    @property
    def y(self) -> np.ndarray:
        """The class label array; one label per instance."""
        return self._y

    def __str__(self):
        s = super().__str__()
        s += f"{self.n_classes} classes:\n"
        s += f"\t{dict(zip(self.classes, self.class_counts))}\n"
        return s


class CombinedDataset(LabelledDataset):
    """A dataset that joins individual datasets together and handles
    labelling differences.
    """

    def __init__(self, *datasets: LabelledDataset):
        self._corpus = "combined"
        self._names = [d.corpus + "_" + n for d in datasets for n in d.names]
        self._feature_names = datasets[0].feature_names
        self._x = np.concatenate([d.x for d in datasets])

        self._partitions = {}
        self._annotation_categories = {}
        self._annotations = {}

        corpora_labels = [d.corpus for d in datasets for _ in d.names]
        self.update_partition("corpora", corpora_labels)

        speakers = [d.corpus + "_" + s for d in datasets for s in d.speakers]
        self.update_speakers(speakers)

        combined_labels = [l for d in datasets for l in d.labels]
        self.update_labels(combined_labels)
        self._y = self.get_annotation_indices("labels")

        common_annotations = set(
            chain.from_iterable(d.annotations.keys() for d in datasets)
        ) - {"labels", "speakers"}
        for annot_name in common_annotations:
            combined_annot = []
            for dataset in datasets:
                annotations = dataset.annotations.get(annot_name)
                if annotations is None:
                    continue
                if type(next(x for x in annotations if x is not None)) == str:
                    annotations = [x if isinstance(x, str) else x for x in annotations]
                    annotations = [dataset.corpus + "_" + str(x) for x in annotations]
                combined_annot.extend(annotations)
            self.update_annotation(annot_name, combined_annot)

    @property
    def corpora(self) -> Sequence[str]:
        """List of corpora in this CombinedDataset."""
        return self.annotation_categories["corpora"]

    @property
    def corpus_indices(self) -> np.ndarray:
        """Indices into corpora list of corresponding corpus for each
        instance.
        """
        return self.get_annotation_indices("corpora")

    @property
    def corpus_counts(self) -> List[int]:
        return self.get_annotation_counts("corpora")

    def corpus_to_idx(self, corpus: str) -> int:
        return self.corpora.index(corpus)

    def get_corpus_split(self, corpus: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a tuple (corpus_idx, other_idx) containing the
        indices of x and y for the specified corpus and all other
        corpora.
        """
        cond = self.corpus_indices == self.corpus_to_idx(corpus)
        corpus_idx = np.nonzero(cond)[0]
        other_idx = np.nonzero(~cond)[0]
        return corpus_idx, other_idx

    def normalise(
        self, normaliser: TransformerMixin = StandardScaler(), scheme: str = "speaker"
    ):

        if scheme == "corpus":
            for corpus in range(len(self.corpora)):
                idx = np.nonzero(self.corpus_indices == corpus)[0]
                flat, slices = inst_to_flat(self.x[idx])
                flat = normaliser.fit_transform(flat)
                self.x[idx] = flat_to_inst(flat, slices)
        else:
            super().normalise(normaliser, scheme)

    def __str__(self) -> str:
        s = super().__str__()
        s += f"{len(self.corpora)} corpora:\n"
        s += f"\t{dict(zip(self.corpora, self.corpus_counts))}\n"
        return s
