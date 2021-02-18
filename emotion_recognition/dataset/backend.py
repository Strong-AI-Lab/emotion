import abc
import json
from collections import Counter
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union

import arff
import netCDF4
import numpy as np
import soundfile

from .binary_arff import decode as decode_arff
from .utils import get_audio_paths, parse_classification_annotations


def _reshape_data_array(x: np.ndarray, slices: np.ndarray) -> np.ndarray:
    """Takes a possibly 2D data array and converts it to either a
    contiguous 2D/3D array or a variable-length 3D array.
    """
    if len(x) == len(slices):
        # 2-D contiguous array
        return x
    elif all(slices == slices[0]):
        # 3-D contiguous array
        assert len(x) % len(slices) == 0
        seq_len = len(x) // len(slices)
        return np.reshape(x, (len(slices), seq_len, x[0].shape[-1]))
    else:
        # 3-D variable length array
        indices = np.cumsum(slices)
        arrs = np.split(x, indices[:-1], axis=0)
        return np.array(arrs, dtype=object)


class DatasetBackend(abc.ABC):
    """Opens the file/directory given by path and reads in the
    relevant data in an implementation specific manner.

    Args:
    -----
    path: pathlike
        The file/directory to read data from.
    """

    @abc.abstractmethod
    def __init__(self, path: Union[PathLike, str]):
        pass

    @property
    def features(self) -> np.ndarray:
        """Feature matrix."""
        return self._features

    @property
    def labels(self) -> Optional[List[str]]:
        """Nominal (string) labels."""
        return self._labels

    @property
    def names(self) -> List[str]:
        """Instance names."""
        return self._names

    @property
    def feature_names(self) -> List[str]:
        """Names of features in feature matrix."""
        return self._feature_names

    @property
    def corpus(self) -> str:
        """Corpus ID."""
        return self._corpus

    _features: np.ndarray = np.empty(0)
    _labels: Optional[List[str]] = None
    _names: List[str] = []
    _feature_names: List[str] = []
    _corpus: str = ''


class NetCDFBackend(DatasetBackend):
    """Backend that reads data from a netCDF4 file in our format, which
    is modified from the format used by the auDeep toolkit.
    """
    def __init__(self, path: Union[PathLike, str]):
        dataset = netCDF4.Dataset(path)
        if not hasattr(dataset, 'corpus'):
            raise AttributeError(
                "Dataset at {} has no corpus metadata.".format(path))
        self._corpus = dataset.corpus

        self._names = [Path(f).stem for f in dataset.variables['filename']]
        feature_dim = json.loads(dataset.feature_dims)[-1]
        self._feature_names = ['feature_{}'.format(i + 1) for i in range(
            dataset.dimensions[feature_dim].size)]

        x = np.array(dataset.variables['features'])
        slices = np.array(dataset.variables['slices'])
        self._features = _reshape_data_array(x, slices)
        if 'label_nominal' in dataset.variables:
            self._labels = list(dataset.variables['label_nominal'])

        dataset.close()


class RawAudioBackend(DatasetBackend):
    """Backend that uses audio clip filepaths from a file and loads the
    audio as raw data.
    """
    def __init__(self, path: Union[PathLike, str]) -> None:
        path = Path(path)
        self.feature_names.append('pcm')

        filepaths = get_audio_paths(path)
        self._features = np.empty(len(filepaths), dtype=object)
        for i, filepath in enumerate(filepaths):
            self.names.append(filepath.stem)
            audio, _ = soundfile.read(filepath, always_2d=True,
                                      dtype='float32')
            self.features[i] = audio

        # We assume the file list is at the root of the dataset directory
        self._corpus = path.parent.stem
        label_file = path.parent / 'labels.csv'
        if label_file.exists():
            self._labels = []
            annotations = parse_classification_annotations(label_file)
            self._names = sorted(x for x in self.names if x in annotations)
            for name in self.names:
                self.labels.append(annotations[name])


class ARFFBackend(DatasetBackend):
    """Backend that loads data from an ARFF (text or binary) file."""
    def __init__(self, path: Union[PathLike, str]) -> None:
        path = Path(path)
        if path.suffix == '.bin':
            with open(path, 'rb') as fid:
                data = decode_arff(fid)
        else:
            with open(path) as fid:
                data = arff.load(fid)

        self._corpus = data['relation']
        self._feature_names = [x[0] for x in data['attributes'][1:-1]]

        counts = Counter([x[0] for x in data['data']])
        self._names = list(counts.keys())

        x = np.array([x[1:-1] for x in data['data']])
        slices = np.array(counts.values())
        self._features = _reshape_data_array(x, slices)
        self._labels = list(dict.fromkeys(x[-1] for x in data['data']).keys())
