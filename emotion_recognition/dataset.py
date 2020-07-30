import json
from collections import Counter
from os import PathLike
from pathlib import Path
from typing import Union

import arff
import netCDF4
import numpy as np
import pandas as pd
import soundfile
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, label_binarize

from emotion_recognition.binary_arff import decode as decode_arff
from emotion_recognition.corpora import corpora


def parse_regression_annotations(file: Union[PathLike, str]):
    """Returns a dict of the form {'name': (v1, v2, v3)}."""
    df = pd.read_csv(file, index_col=0)
    annotations = df.to_dict(orient='index')
    return annotations


def parse_classification_annotations(file: Union[PathLike, str]):
    """Returns a dict of the form {'name': emotion}."""
    df = pd.read_csv(file, index_col=0)
    annotations = df.to_dict()[df.columns[0]]
    return annotations


class LabelledDataset():
    def __init__(self, corpus: str,
                 normaliser=StandardScaler(),
                 normalise_method: str = 'speaker',
                 binarise: bool = False):
        if corpus not in corpora:
            raise NotImplementedError(
                "Corpus {} hasn't been implemented yet.".format(corpus))
        self.corpus = corpus

        self.classes = list(corpora[self.corpus].emotion_map.values())
        self.class_to_int = {c: self.classes.index(c) for c in self.classes}
        self.n_classes = len(self.classes)

        self.n_instances = len(self.names)
        self.n_features = len(self.features)

        self.normaliser = normaliser
        self.normalise_method = normalise_method

        self.speakers = corpora[self.corpus].speakers
        self.n_speakers = len(self.speakers)
        get_speaker = corpora[self.corpus].get_speaker
        self.speaker_indices = np.array(
            [self.speakers.index(get_speaker(n)) for n in self.names],
            dtype=int
        )

        # speaker_group_indices give the group index of each speaker
        speaker_groups = corpora[self.corpus].speaker_groups
        speaker_indices_to_group = np.array([
            i for sp in self.speakers for i in range(len(speaker_groups))
            if sp in speaker_groups[i]
        ])
        self.speaker_group_indices = speaker_indices_to_group[
            self.speaker_indices]

        # gender_indices gives the indices for instances from male and female
        # speakers
        self.gender_indices = {'all': np.arange(self.n_instances)}
        if (corpora[self.corpus].male_speakers
                and corpora[self.corpus].female_speakers):
            self.male_speakers = corpora[self.corpus].male_speakers
            self.female_speakers = corpora[self.corpus].female_speakers
            self.m_indices = np.array([i for i in range(self.n_instances)
                                       if get_speaker(self.names[i])
                                       in self.male_speakers], dtype=int)
            self.f_indices = np.array([i for i in range(self.n_instances)
                                       if get_speaker(self.names[i])
                                       in self.female_speakers], dtype=int)
            self.gender_indices['m'] = self.m_indices
            self.gender_indices['f'] = self.f_indices

        # Subclass-specific create_data() method
        self.create_data()

        # labels contains emotion labels and optionally binary arousal/valence
        # labels, as well as optional per-class binary labels
        self.labels = {'all': self.y}
        if binarise:
            self.binary_y = label_binarize(self.y, np.arange(self.n_classes))
            self.labels.update(
                {c: self.binary_y[:, c] for c in range(self.n_classes)})

            if (corpora[corpus].positive_arousal
                    and corpora[corpus].positive_valence):
                print("Binarising arousal and valence")
                positive_arousal = corpora[corpus].positive_arousal
                arousal_map = np.array([
                    1 if c in positive_arousal else 0
                    for c in self.classes
                ])
                positive_valence = corpora[corpus].positive_valence
                valence_map = np.array([
                    1 if c in positive_valence else 0
                    for c in self.classes
                ])
                self.arousal_y = np.array(arousal_map[self.y.astype(int)],
                                          dtype=np.float32)
                self.valence_y = np.array(valence_map[self.y.astype(int)],
                                          dtype=np.float32)
                self.labels['arousal'] = self.arousal_y
                self.labels['valence'] = self.valence_y

        print('Corpus: {}'.format(corpus))
        print('Classes: {} {}'.format(self.n_classes, tuple(self.classes)))
        print('{} speakers'.format(len(self.speakers)))
        print('Normalisation function: {}'.format(self.normaliser.__class__))
        print('Normalise method: {}'.format(self.normalise_method))
        print()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def create_data(self):
        return NotImplementedError()


class NetCDFDataset(LabelledDataset):
    """A dataset contained in netCDF4 files."""
    def __init__(self, file: Union[PathLike, str],
                 normaliser=StandardScaler(),
                 normalise_method: str = 'speaker',
                 binarise: bool = False):
        self.file = Path(file)
        self.dataset = dataset = netCDF4.Dataset(str(self.file))
        self.names = [Path(f).stem for f in dataset.variables['filename']]
        feature_dim = json.loads(dataset.feature_dims)[-1]
        self.features = ['feature_{}'.format(i + 1) for i in range(
            dataset.dimensions[feature_dim].size)]

        if not hasattr(dataset, 'corpus'):
            raise AttributeError(
                "Dataset at {} has no corpus metadata.".format(str(file)))
        corpus = dataset.corpus
        super().__init__(corpus, normaliser=normaliser,
                         normalise_method=normalise_method, binarise=binarise)

        print('{} instances x {} features'.format(self.n_instances,
                                                  self.n_features))
        counts = np.bincount(self.speaker_indices)
        print("Speaker counts:")
        print(' '.join([format(s, '<5s') for s in self.speakers]))
        print(' '.join([format(x, '<5d') for x in counts]))

        self.dataset.close()
        del self.dataset

    def create_data(self):
        self.x = np.array(self.dataset.variables['features'])
        self.y = np.empty(self.n_instances, dtype=np.float32)

        # TODO: make more general
        annotations = parse_classification_annotations(
            self.file.parent.parent / 'labels.csv')
        for i, name in enumerate(self.names):
            emotion = annotations[name]
            self.y[i] = self.class_to_int[emotion]
        sort = np.argsort(self.names)
        self.x = self.x[sort]
        self.y = self.y[sort]
        self.names = [self.names[i] for i in sort]


class TFRecordDataset(LabelledDataset):
    """A dataset contained in TFRecord files."""
    def __init__(self, file: Union[PathLike, str],
                 normaliser=StandardScaler(),
                 normalise_method: str = 'speaker',
                 binarise: bool = False):
        self.tf_dataset = tf.data.TFRecordDataset([str(file)])
        example = tf.train.Example()
        example.ParseFromString(next(iter(self.tf_dataset)).numpy())
        corpus = example.features.feature['corpus'].bytes_list.value[0].decode()
        self.data_shape = tuple(
            example.features.feature['features_shape'].int64_list.value)
        self.data_dtype = example.features.feature['features_dtype'].bytes_list.value[0].decode()
        super().__init__(corpus, normaliser=normaliser,
                         normalise_method=normalise_method, binarise=binarise)

    def create_data(self):
        self.x = []
        self.y = []
        for item in self.tf_dataset:
            example = tf.train.Example()
            example.ParseFromString(item.numpy())
            data = np.frombuffer(example.features.feature['features'].bytes_list.value[0], dtype=self.dtype)
            data = np.reshape(data, self.data_shape)
            label = example.features.feature['label'].bytes_list.value[0].decode()
            label_int = self.class_to_int[label]
            self.x.append(data)
            self.y.append(label_int)
        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)


class RawDataset(LabelledDataset):
    """A raw audio dataset. Should be in WAV files."""
    def __init__(self, file: Union[PathLike, str],
                 corpus: str,
                 normaliser=StandardScaler(),
                 normalise_method: str = 'speaker',
                 binarise: bool = False):
        self.features = ['pcm']

        self.file = file

        self.names = []
        self.filenames = []
        with open(file) as fid:
            for line in fid:
                filename = line.strip()
                self.filenames.append(filename)
                name = Path(filename).stem
                self.names.append(name)

        super().__init__(corpus, normaliser=normaliser,
                         normalise_method=normalise_method, binarise=binarise)

        print("{} audio files".format(self.n_instances))

        del self.filenames

    def create_data(self):
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
            self.y[i] = self.class_to_int[emotion]


class ArffDataset(LabelledDataset):
    """Represents a dataset from ARFF files."""
    def __init__(self, path: Union[PathLike, str],
                 normaliser=StandardScaler(),
                 normalise_method: str = 'speaker',
                 binarise: bool = False):
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
        super().__init__(corpus, normaliser=normaliser,
                         normalise_method=normalise_method, binarise=binarise)

        del self.raw_data

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def create_data(self):
        self.x = np.empty((self.n_instances, self.n_features),
                          dtype=np.float32)
        self.y = np.empty(self.n_instances, dtype=np.float32)
        for i, inst in enumerate(self.raw_data):
            self.x[i, :] = inst[1:-1]
            self.y[i] = self.class_to_int[inst[-1]]

        if self.normalise_method == 'all':
            self.x = self.normaliser.fit_transform(self.x)
        elif self.normalise_method == 'speaker':
            for sp in range(len(self.speakers)):
                idx = self.speaker_indices == sp
                self.x[idx] = self.normaliser.fit_transform(self.x[idx])


class UtteranceDataset(ArffDataset):
    """Represents a dataset consisting of a single vector per instance."""
    def __init__(self, path: Union[PathLike, str],
                 normaliser=StandardScaler(),
                 normalise_method: str = 'speaker',
                 binarise: bool = False):
        super().__init__(path, normaliser=StandardScaler(),
                         normalise_method='speaker', binarise=binarise)

        print('{} instances x {} features'.format(self.n_instances,
                                                  self.n_features))
        counts = np.bincount(self.speaker_indices)
        print("Speaker counts:")
        print(' '.join([format(s, '<5s') for s in self.speakers]))
        print(' '.join([format(x, '<5d') for x in counts]))


class FrameDataset(ArffDataset):
    """Represents a dataset consisting of a sequence of vectors per instance.
    """
    def __init__(self, path: Union[PathLike, str],
                 normaliser=StandardScaler(),
                 normalise_method: str = 'speaker',
                 binarise: bool = False):
        super().__init__(path, normaliser=normaliser,
                         normalise_method=normalise_method, binarise=binarise)

        names = Counter(self.names)  # Ordered by insertion in Python 3.7+
        self.names = list(names.keys())
        self.n_instances = len(self.names)

        idx = np.cumsum([0] + list(names.values()))
        self.speaker_indices = self.speaker_indices[idx[:-1]]
        if hasattr(self, 'speaker_group_indices'):
            self.speaker_group_indices = self.speaker_group_indices[idx[:-1]]
        self.gender_indices = {'all': np.arange(self.n_instances)}

        self.x = np.array([self.x[idx[i]:idx[i + 1]]
                           for i in range(self.n_instances)], dtype=object)
        self.y = self.y[idx[:-1]]

        self.labels = {'all': self.y}
        if binarise:
            self.labels.update({c: self.binary_y[:, c]
                                for c in range(self.n_classes)})

        print("{} sequences of vectors of size {}".format(self.n_instances,
                                                          self.n_features))
        counts = np.bincount(self.speaker_indices)
        print("Speaker counts:")
        print(' '.join([format(s, '<5s') for s in self.speakers]))
        print(' '.join([format(x, '<5d') for x in counts]))

    def pad_arrays(self, pad: int = 32):
        """Pads each array to the nearest multiple of `pad` greater than the array
        size.
        """
        for i in range(len(self.x)):
            x = self.x[i]
            padding = int(np.ceil(x.shape[0] / pad)) * pad - x.shape[0]
            self.x[i] = np.pad(x, ((0, padding), (0, 0)))
