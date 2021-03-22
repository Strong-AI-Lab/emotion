import abc
from collections import Counter
from pathlib import Path
from typing import List

import arff
import netCDF4
import numpy as np
import pandas as pd
import soundfile

from ..utils import PathOrStr
from .utils import get_audio_paths


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
    def __init__(self, path: PathOrStr):
        pass

    @property
    def features(self) -> np.ndarray:
        """Feature matrix."""
        return self._features

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
    _names: List[str] = []
    _feature_names: List[str] = []
    _corpus: str = ''


class NetCDFBackend(DatasetBackend):
    """Backend that reads data from a netCDF4 file in our format, which
    is modified from the format used by the auDeep toolkit.
    """
    def __init__(self, path: PathOrStr):
        dataset = netCDF4.Dataset(path)
        if not hasattr(dataset, 'corpus'):
            raise AttributeError(f"Dataset at {path} has no corpus metadata.")
        self._corpus = dataset.corpus

        self._names = [Path(f).stem for f in dataset.variables['name']]
        self._feature_names = list(dataset.variables['feature_names'])

        x = np.array(dataset.variables['features'])
        slices = np.array(dataset.variables['slices'])
        self._features = _reshape_data_array(x, slices)
        dataset.close()


class RawAudioBackend(DatasetBackend):
    """Backend that uses audio clip filepaths from a file and loads the
    audio as raw data.
    """
    def __init__(self, path: PathOrStr) -> None:
        path = Path(path)
        self.feature_names.append('pcm')

        filepaths = get_audio_paths(path)
        self._features = np.empty(len(filepaths), dtype=object)
        for i, filepath in enumerate(filepaths):
            self.names.append(filepath.stem)
            audio, _ = soundfile.read(filepath, always_2d=True,
                                      dtype='float32')
            self.features[i] = audio
        self._corpus = path.parent.stem


class ARFFBackend(DatasetBackend):
    """Backend that loads data from an ARFF file."""
    def __init__(self, path: PathOrStr) -> None:
        path = Path(path)
        with open(path) as fid:
            data = arff.load(fid)

        self._corpus = data['relation']
        attr_names = [x[0] for x in data['attributes']]
        self._feature_names = attr_names[1:-1]

        counts = Counter([x[0] for x in data['data']])
        self._names = list(counts.keys())

        x = np.array([x[1:-1] for x in data['data']])
        slices = np.array(counts.values())
        self._features = _reshape_data_array(x, slices)


class CSVBackend(DatasetBackend):
    """Backend that loads data from a CSV file."""
    def __init__(self, path: PathOrStr):
        path = Path(path)
        df = pd.read_csv(path, converters={0: str})

        self._corpus = ''
        self._feature_names = df.columns[1:-1]

        counts = Counter(df.iloc[:, 0])
        self._names = list(counts.keys())

        x = np.array(df.iloc[:, 1:-1])
        slices = np.array(counts.values())
        self._features = _reshape_data_array(x, slices)
