import abc
import json
from collections import Counter
from os import PathLike
from pathlib import Path
from typing import List, Union, Dict, Tuple, Optional

import arff
import netCDF4
import numpy as np
import pandas as pd
import soundfile
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, label_binarize

from .binary_arff import decode as decode_arff
from .corpora import corpora
from .utils import clip_arrays, pad_arrays


def parse_regression_annotations(filename: Union[PathLike, str]) \
        -> Dict[str, Dict[str, float]]:
    """Returns a dict of the form {'name': {'v1': v1, ...}}."""
    df = pd.read_csv(filename, index_col=0)
    annotations = df.to_dict(orient='index')
    return annotations


def parse_classification_annotations(filename: Union[PathLike, str]) \
        -> Dict[str, str]:
    """Returns a dict of the form {'name': emotion}."""
    df = pd.read_csv(filename, index_col=0)
    annotations = df.to_dict()[df.columns[0]]
    return annotations


def write_netcdf_dataset(path: Union[PathLike, str],
                         corpus: str,
                         names: List[str],
                         features: np.ndarray,
                         slices: List[int],
                         annotation_path: Union[PathLike, str],
                         annotation_type: str = 'classification'):
    """Writes a netCDF4 dataset to the given path. The dataset should contain
    features and annotations. Note that the features matrix has to be 2-D, and
    can either be a vector per instance, or a sequence of vectors per instance.
    Also note that this cannot represent the spectrograms in the format
    required by auDeep, since that is a 3-D matrix of one spectrogram per
    instance.

    Args:
    -----
    path: pathlike or str
        The path to write the dataset.
    corpus: str
        The corpus name
    names: list of str
        A list of instance names.
    features: ndarray
        A features matrix of shape (length, n_features).
    slices: list of int
        The size of each slice along axis 0 of features. If there is one vector
        per instance, then this will be all 1's, otherwise will have the length
        of the sequence corresponding to each instance.
    annotation_path: pathlike or str
        The path to an annotation file.
    annotation_type: str
        The type of annotations, one of {regression, classification}.
    """
    dataset = netCDF4.Dataset(path, 'w')
    dataset.createDimension('instance', len(names))
    dataset.createDimension('concat', features.shape[0])
    dataset.createDimension('features', features.shape[1])

    _slices = dataset.createVariable('slices', int, ('instance',))
    _slices[:] = slices

    filename = dataset.createVariable('filename', str, ('instance',))
    filename[:] = np.array(names)

    label_nominal = dataset.createVariable('label_nominal', str, ('instance',))
    if annotation_type == 'regression':
        annotations = parse_regression_annotations(annotation_path)
        keys = next(iter(annotations.values())).keys()
        for k in keys:
            var = dataset.createVariable(k, np.float32, ('instance',))
            var[:] = np.array([annotations[x][k] for x in names])
        dataset.setncattr_string(
            'annotation_vars', json.dumps([k for k in keys]))
    elif annotation_type == 'classification':
        annotations = parse_classification_annotations(annotation_path)
        label_nominal[:] = np.array([annotations[x] for x in names])
        dataset.setncattr_string(
            'annotation_vars', json.dumps(['label_nominal']))

    _features = dataset.createVariable('features', np.float32,
                                       ('concat', 'features'))
    _features[:, :] = features

    dataset.setncattr_string('feature_dims',
                             json.dumps(['concat', 'features']))
    dataset.setncattr_string('corpus', corpus or '')
    dataset.close()


def _make_flat(a: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Flattens an array of variable-length sequences."""
    slices = [x.shape[0] for x in a]
    flat = np.concatenate(a)
    return flat, slices


def _make_ragged(flat: np.ndarray,
                 slices: Union[List[int], np.ndarray]) -> np.ndarray:
    """Returns a list of variable-length sequences."""
    indices = np.cumsum(slices)
    arrs = np.split(flat, indices[:-1], axis=0)
    return arrs


class LabelledDataset(abc.ABC):
    """Abstract class representing a dataset containing discrete labels for
    instances.
    """
    def __init__(self, corpus: str):
        if corpus not in corpora:
            raise NotImplementedError(
                "Corpus {} hasn't been implemented yet.".format(corpus))
        self._corpus = corpus
        self._classes = list(corpora[self.corpus].emotion_map.values())
        self._speakers = corpora[self.corpus].speakers
        self._male_speakers = corpora[self.corpus].male_speakers
        self._female_speakers = corpora[self.corpus].female_speakers

        get_speaker = corpora[self.corpus].get_speaker
        self._speaker_indices = np.array(
            [self.speakers.index(get_speaker(n)) for n in self.names],
            dtype=int
        )

        speaker_groups = corpora[self.corpus].speaker_groups
        speaker_indices_to_group = np.array([
            i for sp in self.speakers for i in range(len(speaker_groups))
            if sp in speaker_groups[i]
        ])
        self._speaker_group_indices = speaker_indices_to_group[
            self.speaker_indices]

        self._gender_indices = {'all': np.arange(self.n_instances)}
        if self.male_speakers and self.female_speakers:
            m_indices = np.array(
                [i for i in range(self.n_instances)
                 if get_speaker(self.names[i]) in self.male_speakers],
                dtype=int
            )
            f_indices = np.array(
                [i for i in range(self.n_instances)
                 if get_speaker(self.names[i]) in self.female_speakers],
                dtype=int
            )
            self._gender_indices['m'] = m_indices
            self._gender_indices['f'] = f_indices

        # Subclass-specific _create_data() method
        self._create_data()

        self._labels = {'all': self.y}
        self._class_counts = np.bincount(self.y.astype(int))
        self._speaker_counts = np.bincount(self.speaker_indices)

        self._print_header()

    def _print_header(self):
        print('Corpus: {}'.format(self.corpus))
        print('{} classes:'.format(self.n_classes))
        print(dict(zip(self.classes, self.class_counts)))
        print('{} instances'.format(self.n_instances))
        print('{} features'.format(self.n_features))
        print('{} speakers:'.format(self.n_speakers))
        print(dict(zip(self.speakers, self.speaker_counts)))
        if self.x.dtype == object:
            lengths = [len(x) for x in self.x]
            print('Sequences:')
            print('min length: {}'.format(np.min(lengths)))
            print('mean length: {}'.format(np.mean(lengths)))
            print('max length: {}'.format(np.max(lengths)))
        print()

    def binarise(self, pos_val: List[str] = [], pos_aro: List[str] = []):
        self.binary_y = label_binarize(self.y, np.arange(self.n_classes))
        self._labels.update(
            {c: self.binary_y[:, c] for c in range(self.n_classes)})

        if pos_aro and pos_val:
            print("Binarising arousal and valence")
            arousal_map = np.array([int(c in pos_aro) for c in self.classes])
            valence_map = np.array([int(c in pos_val) for c in self.classes])
            arousal_y = np.array(arousal_map[self.y.astype(int)],
                                 dtype=np.float32)
            valence_y = np.array(valence_map[self.y.astype(int)],
                                 dtype=np.float32)
            self._labels['arousal'] = arousal_y
            self._labels['valence'] = valence_y

    def normalise(self, normaliser: TransformerMixin = StandardScaler(),
                  scheme: str = 'speaker'):
        """Transforms the X data matrix of this dataset using some
        normalisation method. I think in theory this should be idempotent.
        """
        print("Normalising dataset with scheme '{}' using {}.".format(
            scheme, type(normaliser)))

        if scheme == 'all':
            if self.x.dtype == object:
                flat, slices = _make_flat(self.x)
                flat = normaliser.fit_transform(flat)
                self.x = _make_ragged(flat, slices)
            else:
                self.x = normaliser.fit_transform(self.x)
        elif scheme == 'speaker':
            for sp in range(len(self.speakers)):
                idx = np.nonzero(self.speaker_indices == sp)[0]
                if self.x.dtype == object:
                    flat, slices = _make_flat(self.x[idx])
                    flat = normaliser.fit_transform(flat)
                    self.x[idx] = _make_ragged(flat, slices)
                else:
                    self.x[idx] = normaliser.fit_transform(self.x[idx])

    @property
    def corpus(self) -> str:
        """The corpus this LabelledDataset represents."""
        return self._corpus

    @property
    def classes(self) -> List[str]:
        """A list of emotion class labels."""
        return self._classes

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def class_counts(self) -> np.ndarray:
        return self._class_counts

    @property
    def labels(self) -> Dict[str, np.ndarray]:
        """Mapping from label set to array of numeric labels. The keys
        of the dictionary are {'all', 'arousal', 'valence', ''}
        """
        return self._labels

    @property
    def n_instances(self) -> int:
        return len(self.names)

    @property
    def features(self) -> List[str]:
        return self._features

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def normaliser(self) -> TransformerMixin:
        return self._normaliser

    @property
    def normalise_method(self) -> str:
        return self._normalise_method

    @property
    def speakers(self) -> List[str]:
        return self._speakers

    @property
    def male_speakers(self) -> List[str]:
        return self._male_speakers

    @property
    def female_speakers(self) -> List[str]:
        return self._female_speakers

    @property
    def n_speakers(self) -> int:
        return len(self.speakers)

    @property
    def speaker_counts(self) -> np.ndarray:
        return self._speaker_counts

    @property
    def speaker_indices(self) -> np.ndarray:
        return self._speaker_indices

    @property
    def speaker_group_indices(self) -> np.ndarray:
        return self._speaker_group_indices

    @property
    def gender_indices(self) -> Dict[str, np.ndarray]:
        return self._gender_indices

    @property
    def names(self) -> List[str]:
        return self._names

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    def class_to_int(self, c: str) -> int:
        """Returns the index of the given class label."""
        return self.classes.index(c)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return self.n_instances

    @abc.abstractmethod
    def _create_data(_):
        """Creates the x data array and y label array in an
        implementation specific manner.
        """
        raise NotImplementedError(
            "_create_data() should be implemented by subclasses.")


class SequenceDatasetMixin:
    def pad_arrays(self: LabelledDataset, pad: int = 32):
        """Pads each array to the nearest multiple of `pad` greater than
        the array size. Assumes axis 0 of x is time.
        """
        print("Padding array lengths to nearest multiple of {}.".format(pad))
        pad_arrays(self.x, pad=pad)

    def clip_arrays(self: LabelledDataset, length: int):
        """Clips each array to the specified maximum length."""
        print("Clipping arrays to max length {}.".format(length))
        clip_arrays(self.x, length=length)


class NetCDFDataset(LabelledDataset, SequenceDatasetMixin):
    """A dataset contained in netCDF4 files."""
    def __init__(self, path: Union[PathLike, str]):
        self.dataset = dataset = netCDF4.Dataset(path)
        if not hasattr(dataset, 'corpus'):
            raise AttributeError(
                "Dataset at {} has no corpus metadata.".format(path))

        self._names = [Path(f).stem for f in dataset.variables['filename']]
        feature_dim = json.loads(dataset.feature_dims)[-1]
        self._features = ['feature_{}'.format(i + 1) for i in range(
            dataset.dimensions[feature_dim].size)]

        corpus = dataset.corpus
        super().__init__(corpus)

        self.dataset.close()
        del self.dataset

    def _create_data(self):
        self._x = np.array(self.dataset.variables['features'])
        if len(self._names) != len(self._x):
            slices = self.dataset.variables['slices']
            indices = np.cumsum(slices)
            arrs = np.split(self._x, indices[:-1], axis=0)
            self._x = np.array(arrs)

        labels = self.dataset.variables['label_nominal']
        self._y = np.array([self.class_to_int(x) for x in labels],
                           dtype=np.float32)


class RawDataset(LabelledDataset, SequenceDatasetMixin):
    """A raw audio dataset. Should be in WAV files."""
    def __init__(self, path: Union[PathLike, str], corpus: str):
        self._features = ['pcm']

        self.file = path

        self._names = []
        self.filenames = []
        with open(path) as fid:
            for line in fid:
                filename = line.strip()
                self.filenames.append(filename)
                name = Path(filename).stem
                self._names.append(name)

        super().__init__(corpus)

        print("{} audio files".format(self.n_instances))

        del self.filenames

    def _create_data(self):
        self._x = np.empty(self.n_instances, dtype=object)
        self._y = np.empty(self.n_instances, dtype=np.float32)
        for i, filename in enumerate(self.filenames):
            audio, _ = soundfile.read(filename, dtype=np.float32)
            audio = np.expand_dims(audio, axis=1)
            self._x[i] = audio

            annotations = parse_classification_annotations(
                Path(self.file).parent / 'labels.csv')
            name = Path(filename).stem
            emotion = annotations[name]
            self._y[i] = self.class_to_int(emotion)


class UtteranceARFFDataset(LabelledDataset):
    """Represents an ARFF dataset consisting of a single vector per
    instance.
    """
    def __init__(self, path: Union[PathLike, str]):
        path = Path(path)
        if path.suffix == '.bin':
            with open(path, 'rb') as fid:
                data = decode_arff(fid)
        else:
            with open(path) as fid:
                data = arff.load(fid)

        self.raw_data = data['data']
        self._names = [x[0] for x in self.raw_data]
        self._features = [x[0] for x in data['attributes'][1:-1]]

        corpus = data['relation']
        super().__init__(corpus)

        del self.raw_data

    def _create_data(self):
        self._x = np.array([x[1:-1] for x in self.raw_data])
        self._y = np.array([self.class_to_int(x[-1]) for x in self.raw_data],
                           dtype=np.float32)


class SequenceARFFDataset(LabelledDataset, SequenceDatasetMixin):
    """Represents a dataset consisting of a sequence of vectors per
    instance.
    """
    def __init__(self, path: Union[PathLike, str]):
        path = Path(path)
        if path.suffix == '.bin':
            with open(path, 'rb') as fid:
                data = decode_arff(fid)
        else:
            with open(path) as fid:
                data = arff.load(fid)

        self.raw_data = data['data']
        names = Counter(self.names)  # Ordered by insertion in Python 3.7+
        self._names = list(names.keys())
        self.slices = np.array(names.values())
        self._features = [x[0] for x in data['attributes'][1:-1]]

        corpus = data['relation']
        super().__init__(corpus)

        del self.raw_data, self.slices

    def _create_data(self):
        self._x = np.array([x[1:-1] for x in self.raw_data])
        indices = np.cumsum(self.slices)
        arrs = np.split(self._x, indices[:-1], axis=0)
        self._x = np.array(arrs)

        labels = dict.fromkeys(x[-1] for x in self.raw_data).keys()
        self._y = np.array([self.class_to_int(x) for x in labels],
                           dtype=np.float32)


class CombinedDataset(LabelledDataset, SequenceDatasetMixin):
    """A dataset that joins individual corpus datasets together and
    handles labelling differences.
    """
    def __init__(self, *datasets: LabelledDataset,
                 labels: Optional[List[str]] = None):
        self._x = np.concatenate([x.x for x in datasets])
        self._corpora = [x.corpus for x in datasets]
        self._corpus = 'combined'
        self._names = [d.corpus + '_' + n for d in datasets for n in d.names]
        self._features = datasets[0].features
        self._speakers = self._corpora

        all_labels = set(c for d in datasets for c in d.classes)
        self._classes = sorted(all_labels)

        sizes = [len(x.x) for x in datasets]
        self._speaker_indices = np.repeat(np.arange(len(datasets)), sizes)

        str_labels = [d.classes[int(i)] for d in datasets for i in d.y]
        if labels:
            drop_labels = all_labels - labels
            keep_idx = [i for i, x in enumerate(str_labels)
                        if x not in drop_labels]
            self._x = self._x[keep_idx]
            self._speaker_indices = self._speaker_indices[keep_idx]
            self._classes = sorted(labels)
            str_labels = [x for x in str_labels if x not in drop_labels]
        self._y = np.array([self._classes.index(y) for y in str_labels],
                           dtype=np.float32)
        self._speaker_group_indices = self._speaker_indices

        self._labels = {'all': self.y}

        self._print_header()

    def _create_data(_):
        pass

    @property
    def corpora(self) -> List[str]:
        return self._corpora

    def corpus_to_idx(self, corpus: str) -> int:
        return self.corpora.index(corpus)

    def get_corpus_split(self, corpus: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a tuple (corpus_idx, other_idx) containing the
        indices of x and y for the specified corpus and all other
        corpora.
        """
        cond = self.speaker_indices == self.corpus_to_idx(corpus)
        corpus_idx = np.nonzero(cond)[0]
        other_idx = np.nonzero(~cond)[0]
        return corpus_idx, other_idx
