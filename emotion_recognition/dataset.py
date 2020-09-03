import json
from abc import ABC, abstractmethod
from collections import Counter
from os import PathLike
from pathlib import Path
from typing import List, Union, Dict, Tuple

import arff
import netCDF4
import numpy as np
import pandas as pd
import soundfile
import tensorflow as tf
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, label_binarize

from emotion_recognition.binary_arff import decode as decode_arff
from emotion_recognition.corpora import corpora


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


def _make_flat(a: np.ndarray) -> np.ndarray:
    slices = np.array([x.shape[0] for x in a])
    flat = np.concatenate(a)
    return flat, slices


def _make_ragged(flat: np.ndarray, slices: np.ndarray) -> np.ndarray:
    indices = np.cumsum(slices)
    arrs = np.split(flat, indices[:-1], axis=0)
    ragged = np.array(arrs)
    return ragged


class LabelledDataset(ABC):
    def __init__(self, corpus: str):
        if corpus not in corpora:
            raise NotImplementedError(
                "Corpus {} hasn't been implemented yet.".format(corpus))
        self._corpus = corpus

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

        self._print_header()

    def _print_header(self):
        print('Corpus: {}'.format(self.corpus))
        print('{} classes: {}'.format(self.n_classes, tuple(self.classes)))
        print('{} instances'.format(self.n_instances))
        print('{} features'.format(self.n_features))
        print('{} speakers:'.format(len(self.speakers)))
        counts = np.bincount(self.speaker_indices)
        print(' '.join([format(s, '<5s') for s in self.speakers]))
        print(' '.join([format(x, '<5d') for x in counts]))
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
                idx = self.speaker_indices == sp
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
        return list(corpora[self.corpus].emotion_map.values())

    @property
    def labels(self) -> Dict[str, np.ndarray]:
        """Mapping from label set to array of numeric labels. The keys of the
        dictionary are {'all', 'arousal', 'valence', ''}
        """
        return self._labels

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_instances(self) -> int:
        return len(self.names)

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
        return corpora[self.corpus].speakers

    @property
    def male_speakers(self) -> List[str]:
        return corpora[self.corpus].male_speakers

    @property
    def female_speakers(self) -> List[str]:
        return corpora[self.corpus].female_speakers

    @property
    def n_speakers(self) -> int:
        return len(self.speakers)

    @property
    def speaker_indices(self) -> np.ndarray:
        return self._speaker_indices

    @property
    def speaker_group_indices(self) -> np.ndarray:
        return self._speaker_group_indices

    @property
    def gender_indices(self) -> Dict[str, np.ndarray]:
        return self._gender_indices

    def class_to_int(self, c: str) -> int:
        """Returns the index of the given class label."""
        return self.classes.index(c)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[idx], self.y[idx]

    @abstractmethod
    def _create_data(self):
        """Creates the x data array and y label array in an implementation
        specific manner.
        """
        raise NotImplementedError(
            "_create_data() should be implemented by subclasses.")


class NetCDFDataset(LabelledDataset):
    """A dataset contained in netCDF4 files."""
    def __init__(self, path: Union[PathLike, str]):
        self.dataset = dataset = netCDF4.Dataset(path)
        if not hasattr(dataset, 'corpus'):
            raise AttributeError(
                "Dataset at {} has no corpus metadata.".format(path))

        self.names = [Path(f).stem for f in dataset.variables['filename']]
        feature_dim = json.loads(dataset.feature_dims)[-1]
        self.features = ['feature_{}'.format(i + 1) for i in range(
            dataset.dimensions[feature_dim].size)]

        corpus = dataset.corpus
        super().__init__(corpus)

        self.dataset.close()
        del self.dataset

    def _create_data(self):
        self.x = np.array(self.dataset.variables['features'])
        if len(self.names) != len(self.x):
            slices = self.dataset.variables['slices']
            indices = np.cumsum(slices)
            arrs = np.split(self.x, indices[:-1], axis=0)
            self.x = np.array(arrs)

        labels = self.dataset.variables['label_nominal']
        self.y = np.array([self.class_to_int(x) for x in labels],
                          dtype=np.float32)


class TFRecordDataset(LabelledDataset):
    """A dataset contained in TFRecord files."""
    def __init__(self, file: Union[PathLike, str]):
        self.tf_dataset = tf.data.TFRecordDataset([str(file)])
        example = tf.train.Example()
        example.ParseFromString(next(iter(self.tf_dataset)).numpy())
        corpus = example.features.feature['corpus'].bytes_list.value[0].decode()
        self.data_shape = tuple(
            example.features.feature['features_shape'].int64_list.value)
        self.data_dtype = example.features.feature['features_dtype'].bytes_list.value[0].decode()
        super().__init__(corpus)

    def _create_data(self):
        self.x = []
        self.y = []
        for item in self.tf_dataset:
            example = tf.train.Example()
            example.ParseFromString(item.numpy())
            data = np.frombuffer(
                example.features.feature['features'].bytes_list.value[0],
                dtype=self.dtype
            )
            data = np.reshape(data, self.data_shape)
            label = example.features.feature['label'].bytes_list.value[0].decode()
            label_int = self.class_to_int(label)
            self.x.append(data)
            self.y.append(label_int)
        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)


class RawDataset(LabelledDataset):
    """A raw audio dataset. Should be in WAV files."""
    def __init__(self, path: Union[PathLike, str], corpus: str):
        self.features = ['pcm']

        self.file = path

        self.names = []
        self.filenames = []
        with open(path) as fid:
            for line in fid:
                filename = line.strip()
                self.filenames.append(filename)
                name = Path(filename).stem
                self.names.append(name)

        super().__init__(corpus)

        print("{} audio files".format(self.n_instances))

        del self.filenames

    def _create_data(self):
        self.x = np.empty(self.n_instances, dtype=object)
        self.y = np.empty(self.n_instances, dtype=np.float32)
        for i, filename in enumerate(self.filenames):
            audio, sr = soundfile.read(filename, dtype=np.float32)
            audio = np.expand_dims(audio, axis=1)
            self.x[i] = audio

            annotations = parse_classification_annotations(
                Path(self.file).parent / 'labels.csv')
            name = Path(filename).stem
            emotion = annotations[name]
            self.y[i] = self.class_to_int(emotion)


class UtteranceARFFDataset(LabelledDataset):
    """Represents an ARFF dataset consisting of a single vector per instance.
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
        self.names = [x[0] for x in self.raw_data]
        self.features = [x[0] for x in data['attributes'][1:-1]]

        corpus = data['relation']
        super().__init__(corpus)

        del self.raw_data

    def _create_data(self):
        self.x = np.array([x[1:-1] for x in self.raw_data])
        self.y = np.array([self.class_to_int(x[-1]) for x in self.raw_data],
                          dtype=np.float32)


class FrameARFFDataset(LabelledDataset):
    """Represents a dataset consisting of a sequence of vectors per instance.
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
        self.names = list(names.keys())
        self.slices = np.array(names.values())
        self.features = [x[0] for x in data['attributes'][1:-1]]

        corpus = data['relation']
        super().__init__(corpus)

        del self.raw_data, self.slices

    def _create_data(self):
        self.x = np.array([x[1:-1] for x in self.raw_data])
        indices = np.cumsum(self.slices)
        arrs = np.split(self.x, indices[:-1], axis=0)
        self.x = np.array(arrs)

        labels = dict.fromkeys(x[-1] for x in self.raw_data).keys()
        self.y = np.array([self.class_to_int(x) for x in labels],
                          dtype=np.float32)

    def pad_arrays(self, pad: int = 32):
        """Pads each array to the nearest multiple of `pad` greater than the array
        size.
        """
        for i in range(len(self.x)):
            x = self.x[i]
            padding = int(np.ceil(x.shape[0] / pad)) * pad - x.shape[0]
            self.x[i] = np.pad(x, ((0, padding), (0, 0)))
