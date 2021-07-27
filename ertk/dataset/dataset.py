import logging
import os
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
    Type,
    TypeVar,
    Union,
)

import numpy as np
import yaml
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

from ..utils import (
    PathOrStr,
    clip_arrays,
    frame_arrays,
    group_transform,
    pad_arrays,
    transpose_time,
)
from .annotation import read_annotations
from .features import find_features_file, read_features
from .utils import get_audio_paths

_D = TypeVar("_D", bound="Dataset")


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

    Args:
    -----
    corpus_info: Pathlike or str, optional
        Path to corpus info in YAML format.
    features: Pathlike or str, optional
        Path to features file, or unique name of features in corpus
        features directory.
    subset: str, optional
        The subset of instances to use.
    """

    _partitions: Set[str]
    _annotations: Dict[str, Dict[str, Any]]
    _corpus: str = ""
    _names: List[str]
    _feature_names: List[str]
    _x: np.ndarray
    _subset: str = ""
    _subsets: Dict[str, Collection[str]]

    _default_subset: str = ""
    _features_path: Path
    _features_dir: Path

    def __init__(
        self,
        corpus_info: Optional[PathOrStr] = None,
        features: Optional[PathOrStr] = None,
        subset: str = "default",
    ):
        self._partitions = set()
        self._annotations = {}
        self._subsets = {}
        self._names = []
        self._feature_names = []
        self._x = np.empty((0, 0), dtype=np.float32)

        if corpus_info is not None:
            self.init_corpus_info(corpus_info)
            self.use_subset(subset)
        if features is not None:
            self.update_features(features)

    def init_corpus_info(self, path: PathOrStr):
        """Initialise corpus metadata from YAML.

        Args:
        -----
        path: os.Pathlike or str
            The path to a YAML file containing corpus metadata.
        """
        path = Path(path)
        with open(path) as fid:
            doc = yaml.safe_load(fid)

        if self.corpus == "":
            self._corpus = next(iter(doc))
        corpus_info = doc[self.corpus]

        self._features_dir = path.parent / corpus_info["features_dir"]

        if "subsets" in corpus_info:
            for subset, subset_info in corpus_info["subsets"].items():
                subset_info = corpus_info["subsets"][subset]
                if subset == "default":
                    self._default_subset = subset_info
                    continue
                clips_file = path.parent / f"{subset_info['clips']}.txt"
                self.subsets[subset] = {x.stem for x in get_audio_paths(clips_file)}
        if corpus_info.get("partitions", None):
            for partition in corpus_info["partitions"]:
                self.update_annotation(
                    partition, path.parent / f"{partition}.csv", dtype=str
                )
        if corpus_info.get("annotations", None):
            for annotation in corpus_info["annotations"]:
                self.update_annotation(annotation, path.parent / f"{annotation}.csv")

    def _verify_annotations(self, annot_name: Optional[str] = None):
        def verify_annotation(annot_name):
            annotations = self.annotations[annot_name]
            missing = set(self.names) - annotations.keys()
            if len(missing) > 0:
                raise ValueError(
                    f"Incomplete annotation {annot_name}. These names are missing:\n"
                    f"{missing}"
                )

        if annot_name is not None:
            verify_annotation(annot_name)
        else:
            for annot_name in self.annotations:
                verify_annotation(annot_name)

    def update_features(self, features: PathOrStr, subset: Optional[str] = None):
        """Update the features matrix, instance names and feature names
        for this dataset.

        Args:
        -----
        features: os.PathLike or str
            Path to a set of features or unique name of features in
            corpus features dir.
        subset: str, optional
            The subset to select.
        """
        features = Path(features)
        if len(features.suffix) == 0:
            features = find_features_file(self._features_dir.glob(f"{features}.*"))
        self._features = features.stem
        data = read_features(features)
        self._feature_names = data.feature_names
        self._x = data.features
        self._names = data.names
        if not subset:
            subset = self.subset
        self.use_subset(subset)

    def use_subset(self, subset: str = "default"):
        """Use a different subset of the instances.

        Args:
        -----
        subset: str
            Name of subset to use. Default is "default" which uses the
            default subset specified in corpus_info.
        """
        if len(self._subsets) == 0:
            self._subset = "all"
        else:
            if subset == "default":
                subset = self._default_subset
            self._subset = subset
            self.remove_instances(keep=self.subsets[subset])
        self._verify_annotations()

    def update_annotation(
        self,
        annot_name: str,
        annotations: Union[PathOrStr, Mapping[str, Any], Sequence[Any]],
        dtype: Optional[Type] = None,
    ):
        """Add or update an annotation.

        Args:
        -----
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
        if isinstance(next(iter(annot_dict.values())), str):
            self.partitions.add(annot_name)
        self._verify_annotations(annot_name)

    def remove_annotation(self, annot_name: str):
        """Removes a set of annotations from the dataset.

        Args:
        -----
        annot_name: str
            Annotation name.
        """
        del self.annotations[annot_name]
        self.partitions.difference_update({annot_name})

    def get_annotations(self, annot_name: str) -> List[Any]:
        """Get a list of annotations for each instance currently in the
        dataset.

        Args:
        -----
        annot_name: str
            Annotation name.

        Returns:
        --------
        A list of values, one for each instance in the datset, in the
        same order they appear in names and x.
        """
        return [self.annotations[annot_name][x] for x in self.names]

    def get_group_indices(self, annot_name: str) -> np.ndarray:
        """Gets the group indices (i.e. indices into the groups array)
        for a given partition.

        Args:
        -----
        annot_name: str
            The partition name.

        Returns:
        --------
        A NumPy array of group indices for each instance in the dataset.
        """
        _, idx = np.unique(self.get_annotations(annot_name), return_inverse=True)
        return idx

    def get_group_counts(self, annot_name: str) -> np.ndarray:
        """Get group counts for a partition.

        Args:
        -----
        annot_name: str
            The partition name.

        Returns:
        --------
        A NumPy array of counts for the corresponding group in this
        partition.
        """
        _, counts = np.unique(self.get_annotations(annot_name), return_counts=True)
        return counts

    def get_group_names(self, annot_name: str) -> List[str]:
        """Get the names of groups in a partition.

        Args:
        -----
        annot_name: str
            Annotation name.
        """
        return list(np.unique(self.get_annotations(annot_name)))

    def update_speakers(
        self, speakers: Union[PathOrStr, Mapping[str, str], Sequence[str]]
    ):
        """Update the speakers for this dataset.

        Args:
        -----
        speakers: Path, str, dict or list
            If Path or str, read speaker info from CSV at given path. If
            dict, use speaker dict directly. If list, each speakers[i]
            is the corresponding speaker for instance i.
        """
        self.update_annotation("speaker", speakers, dtype=str)

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
        self._verify_annotations()

    def copy(self: _D) -> _D:
        import copy

        new_dataset = type(self)()
        new_dataset._annotations = copy.deepcopy(self._annotations)
        new_dataset._corpus = self._corpus
        new_dataset._feature_names = self._feature_names.copy()
        new_dataset._names = self._names.copy()
        new_dataset._partitions = copy.deepcopy(self._partitions)
        new_dataset._x = self._x.copy()
        return new_dataset

    def normalise(
        self, partition: str, normaliser: TransformerMixin = StandardScaler()
    ):
        """Transforms the data matrix of this dataset in-place using
        some (offline) normalisation method.

        Args:
        -----
        normaliser:
            The transform to apply. Must implement fit_transform().
        partition: str
            The partition to apply in a per-group fashion. If "all",
            then perform global normalisation on all the data. Default
            is "speaker" to do per-speaker normalisation.
        """

        if partition == "all":
            groups = np.zeros(len(self.x), dtype=int)
        else:
            groups = self.get_group_indices(partition)
        group_transform(self.x, groups, normaliser, inplace=True)

    def pad_arrays(self, pad: int = 32):
        """Pads each array to the nearest multiple of `pad` greater than
        the array size. Assumes axis 0 of x is time.
        """
        logging.info(f"Padding array lengths to nearest multiple of {pad}.")
        pad_arrays(self.x, pad=pad)

    def clip_arrays(self, length: int):
        """Clips each array to the specified maximum length."""
        logging.info(f"Clipping arrays to max length {length}.")
        clip_arrays(self.x, length=length)

    def frame_arrays(
        self,
        frame_size: int = 640,
        frame_shift: int = 160,
        num_frames: Optional[int] = None,
    ):
        """Create a sequence of frames from the raw signal."""
        logging.info(f"Framing arrays with size {frame_size} and shift {frame_shift}.")
        self._x = frame_arrays(
            self._x,
            frame_size=frame_size,
            frame_shift=frame_shift,
            num_frames=num_frames,
        )

    def transpose_time(self):
        """Transpose the time and feature axis of each instance."""
        logging.info("Transposing time and feature axis of data.")
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
        """Mapping of instance partitions."""
        return self._partitions

    @property
    def part_names(self) -> List[str]:
        return list(self.partitions)

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
    def subsets(self) -> Dict[str, Collection[str]]:
        """List of subsets"""
        return self._subsets

    @property
    def x(self) -> np.ndarray:
        """The data matrix."""
        return self._x

    def __len__(self) -> int:
        return self.n_instances

    def __str__(self):
        s = f"Corpus: {self.corpus}\n"
        for part in self.partitions:
            group_names = self.get_group_names(part)
            s += f"partition '{part}'' ({len(group_names)} groups):\n"
            s += f"\t{dict(zip(group_names, self.get_group_counts(part)))}\n"
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

    def update_labels(self, labels: Union[PathOrStr, Mapping[str, str], Sequence[str]]):
        self.update_annotation("label", labels, dtype=str)

    def map_classes(self, mapping: Mapping[str, str]):
        """Modifies classses based on the mapping in map. Keys not
        corresponding to classes are ignored. The new classes will be
        sorted lexicographically.
        """
        new_labels = [mapping.get(x, x) for x in self.labels]
        self.update_labels(new_labels)

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
    def classes(self) -> List[str]:
        """A list of emotion class labels."""
        return self.get_group_names("label")

    @property
    def n_classes(self) -> int:
        """Total number of emotion classes."""
        return len(self.classes)

    @property
    def class_counts(self) -> np.ndarray:
        """Number of instances for each class."""
        return self.get_group_counts("label")

    @property
    def labels(self) -> List[str]:
        """Mapping from instance to label."""
        return self.get_annotations("label")

    @property
    def y(self) -> np.ndarray:
        """The class label array; one label per instance."""
        return self.get_group_indices("label")


class CombinedDataset(LabelledDataset):
    """A dataset that joins individual datasets together and handles
    labelling differences.
    """

    def __init__(self, *datasets: LabelledDataset):
        super().__init__()

        logging.info("Combining " + ", ".join(d.corpus for d in datasets))

        self._corpus = "combined"
        self._names = [d.corpus + "_" + n for d in datasets for n in d.names]
        self._feature_names = datasets[0].feature_names
        self._features = datasets[0]._features
        self._x = np.concatenate([d.x for d in datasets])

        self.update_annotation(
            "corpus", [d.corpus for d in datasets for _ in d.names], dtype=str
        )
        self.update_speakers([d.corpus + "_" + s for d in datasets for s in d.speakers])

        common_annotations = set(chain(*(d.annotations for d in datasets)))
        common_annotations -= {"speaker"}  # assumed to be per-dataset
        for annot_name in common_annotations:
            combined_annot = {}
            for dataset in datasets:
                annotations = dataset.annotations.get(annot_name, {})
                combined_annot.update(
                    {dataset.corpus + "_" + k: v for k, v in annotations.items()}
                )
            self.update_annotation(annot_name, combined_annot)

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

    @property
    def corpus_counts(self) -> List[int]:
        return self.get_group_counts("corpus")


def load_multiple(corpus_files: Iterable[PathOrStr], features: str) -> CombinedDataset:
    corpus_files = list(corpus_files)
    if len(corpus_files) == 0:
        raise RuntimeError("No corpus metadata files given.")
    datasets = []
    for file in corpus_files:
        dataset = LabelledDataset(file)
        logging.info(f"Loading {dataset.corpus}")
        dataset.update_features(features)
        datasets.append(dataset)
    return CombinedDataset(*datasets)
