"""Dataset features backend."""

import csv
import itertools
import logging
import typing
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sized,
    Type,
    Union,
)

import arff
import librosa
import netCDF4
import numpy as np
import pandas as pd
import soundfile
from joblib import delayed

from ertk.dataset.utils import get_audio_paths, write_filelist
from ertk.utils import (
    PathOrStr,
    TqdmParallel,
    filter_kwargs,
    flat_to_inst,
    inst_to_flat,
    make_array_array,
)

__all__ = [
    "read_features",
    "read_features_iterable",
    "write_features",
    "register_format",
    "FeaturesData",
    "SequentialReader",
    "RandomAccessReader",
    "SequentialWriter",
    "NetCDFReader",
    "RawAudioReader",
    "ARFFSequentialReader",
    "CSVSequentialReader",
]

logger = logging.getLogger(__name__)


class FeaturesData:
    """Dataset features backend.

    Parameters
    ----------
    features: ndarray
        Features matrix. May be "flat" or per-instance. If flat,
        `slices` must also be present.
    names: list of str
        List of instance names.
    corpus: str
        Corpus for this dataset.
    feature_names: list of str, optional
        List of feature names. If not given, the features will me named
        feature1, feature2, etc.
    slices: list or ndarray:
        Slices corresponding to "flat" `features` matrix.
    dataset: DatasetBackend
        Other backend instance to copy attributes from. Takes priority
        over other passed args.
    """

    def __init__(
        self,
        features: Union[List[np.ndarray], np.ndarray],
        names: List[str],
        corpus: str = "",
        feature_names: Optional[List[str]] = None,
        slices: Union[np.ndarray, List[int], None] = None,
    ):
        self._corpus = corpus
        self._names = list(names)
        if isinstance(features, list):
            features = make_array_array(features)
        self._features = features
        if feature_names is None:
            feature_names = []
        self._feature_names = list(feature_names)

        if slices is not None:
            self._slices = np.array(slices)
            self._flat = features
            self._features = flat_to_inst(self.flat, slices)
        else:
            self.make_contiguous()

        n_features = self.features[0].shape[-1]
        if len(self.feature_names) == 0:
            self._feature_names = [f"feature{i + 1}" for i in range(n_features)]
        elif len(self.feature_names) != n_features:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries but there "
                f"are {n_features} features"
            )

    def make_contiguous(self):
        """Makes `self.features` a contiguous array that represents
        views of `self.flat`.
        """
        self._flat, self._slices = inst_to_flat(self.features)
        self._features = flat_to_inst(self.flat, self.slices)

    def write(self, path: PathOrStr, **kwargs) -> None:
        """Write features to any file format."""

        write_features(
            path,
            self.features,
            names=self.names,
            corpus=self.corpus,
            feature_names=self.feature_names,
            **kwargs,
        )

    @property
    def features(self) -> np.ndarray:
        """Instance features matrix. This may be 2D/3D contiguous or
        variable length 3D.
        """
        return self._features

    @property
    def flat(self) -> np.ndarray:
        """2D "flat" features matrix. This has shape (N, F), where
        N = sum(slices), F is the number of features.
        """
        return self._flat

    @property
    def slices(self) -> np.ndarray:
        """Slices corresponding to flat features matrix. It should be
        the case that `features = flat_to_inst(flat, slices)` and
        `flat, slices = inst_to_flat(features)`.
        """
        return self._slices

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

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index) -> np.ndarray:
        return self.features[index]


def read_arff_mem(path: PathOrStr, label: bool = False) -> FeaturesData:
    path = Path(path)
    with open(path) as fid:
        data = arff.load(fid)

    attr_names = [x[0] for x in data["attributes"]]
    counts = Counter([x[0] for x in data["data"]])
    idx = slice(1, -1) if label else slice(1, None)

    return FeaturesData(
        corpus=data["relation"],
        names=list(counts.keys()),
        features=np.array([x[idx] for x in data["data"]]),
        slices=np.array(list(counts.values())),
        feature_names=attr_names[idx],
    )


def read_csv_mem(
    path: PathOrStr, header: bool = True, label: bool = False
) -> FeaturesData:
    df = pd.read_csv(path, header=0 if header else None, converters={0: str})
    counts: typing.Counter[str] = Counter(df.iloc[:, 0])
    idx = slice(1, -1) if label else slice(1, None)

    return FeaturesData(
        names=list(counts.keys()),
        features=np.array(df.iloc[:, idx]),
        slices=np.array(list(counts.values())),
        feature_names=list(df.columns[idx]),
    )


def read_netcdf_mem(path: PathOrStr) -> FeaturesData:
    reader = NetCDFReader(path)
    features = reader[:]
    reader.close()
    return FeaturesData(features, reader.names, reader.corpus, reader.feature_names)


def _read_audio_file(path, sample_rate: int = 16000):
    warnings.filterwarnings("ignore", message="PySoundFile", category=UserWarning)
    audio, _ = librosa.load(path, sr=sample_rate, mono=True, res_type="kaiser_fast")
    return np.expand_dims(audio, -1)


def read_raw_mem(path: PathOrStr, sample_rate: int = 16000) -> FeaturesData:
    logger.info(f"Reading raw audio with sample_rate: {sample_rate}")
    path = Path(path)
    filepaths = get_audio_paths(path)
    _audio = TqdmParallel(len(filepaths), desc="Reading audio", leave=False, n_jobs=-1)(
        delayed(_read_audio_file)(filepath, sample_rate) for filepath in filepaths
    )

    return FeaturesData(
        corpus=path.parent.stem,
        names=[x.stem for x in filepaths],
        features=np.concatenate(_audio),
        slices=[len(x) for x in _audio],
        feature_names=["pcm"],
    )


class SequentialReader(Iterable[np.ndarray], Sized, ABC):
    """Base class for readers that read features sequentially."""

    feature_names: List[str]
    names: List[str]
    corpus: str = ""

    def __init__(self, path: PathOrStr, **kwargs) -> None:
        self.path = path
        self.open()

    def __len__(self):
        return len(self.names)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self.read()

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def read(self) -> Iterable[np.ndarray]:
        raise NotImplementedError()


class RandomAccessReader(SequentialReader, ABC):
    """Base class for readers that can access features by index."""

    @abstractmethod
    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray]
    ) -> np.ndarray:
        raise NotImplementedError()

    def close(self) -> None:
        pass

    def read(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self[i]


class NetCDFReader(RandomAccessReader):
    """Read features from NetCDF4 file."""

    _is_2d: bool = False
    _is_3d: bool = False
    _is_vlen: bool = False

    def open(self) -> None:
        dataset = netCDF4.Dataset(self.path)
        self.corpus = getattr(dataset, "corpus", "")
        self.names = list(dataset.variables["name"])
        self.slices = np.asarray(dataset.variables["slices"])
        data_version = getattr(dataset, "ertk_data_version", "1")
        if data_version == "1":
            self.feature_names = list(dataset.variables["feature_names"])
            self._feat_var = "features"
        else:
            self.feature_names = list(dataset.variables["feature"])
            self._feat_var = "data"
        unique = np.unique(self.slices)
        if len(unique) == 1:
            if unique[0] == 1:
                self._is_2d = True
            else:
                self._is_3d = True
        else:
            self._is_vlen = True
        self.index = np.r_[0, np.cumsum(self.slices)]
        self.dataset = dataset

    def close(self) -> None:
        self.dataset.close()

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray]
    ) -> np.ndarray:
        feats_var = self.dataset.variables[self._feat_var]
        if self._is_2d:
            # Shortcut when we don't have to manipulate indices
            return np.asarray(feats_var[index])
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("Non-contiguous slice (step != 1).")
            if start >= stop:
                raise ValueError("Empty slice.")
            istart = self.index[start]
            iend = self.index[stop]
            arr = np.asarray(feats_var[istart:iend])
            return flat_to_inst(arr, self.slices[start:stop])
        if isinstance(index, (list, np.ndarray)):
            _idx = np.array(index, dtype=np.int64)
            if _idx.ndim != 1:
                raise TypeError("If index is list/array, must be 1D.")
            items = [self.__getitem__(x) for x in _idx]
            if self._is_3d:
                return np.stack(items)
            else:
                return make_array_array(items)
        if isinstance(index, int):
            start, end = self.index[index], self.index[index + 1]
            return np.asarray(feats_var[start:end])
        raise NotImplementedError()


class RawAudioReader(RandomAccessReader):
    """Read raw audio files."""

    def __init__(self, path: PathOrStr, sample_rate: int = 16000, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self.sample_rate = sample_rate

    def open(self) -> None:
        self.filepaths = np.array(get_audio_paths(self.path), dtype=object)
        self.names = [x.stem for x in self.filepaths]
        self.feature_names = ["pcm"]

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray]
    ) -> np.ndarray:
        if isinstance(index, int):
            return _read_audio_file(self.filepaths[index], self.sample_rate)
        return make_array_array(
            [
                _read_audio_file(filepath, self.sample_rate)
                for filepath in self.filepaths[index]
            ]
        )


class CSVSequentialReader(SequentialReader):
    """Read features from CSV file."""

    def open(self) -> None:
        with open(self.path) as fid:
            reader = csv.reader(fid)
            _, *feature_names = next(reader)
            self.feature_names = feature_names
            _names = [row[0] for row in reader]
            counts: typing.Counter[str] = Counter(_names)
            self.names = list(counts.keys())
            self.slices = np.array(list(counts.values()))

    def read(self) -> Iterator[np.ndarray]:
        with open(self.path) as fid:
            reader = csv.reader(fid)
            next(reader)  # Skip header
            for length in self.slices:
                yield np.array(
                    [x[1:] for x in itertools.islice(reader, length)]
                ).squeeze(0)


class ARFFSequentialReader(SequentialReader):
    """Read features from ARFF file."""

    def open(self) -> None:
        self.feature_names = []
        with open(self.path) as fid:
            for i, line in enumerate(fid):
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.lower().startswith("@relation"):
                    self.corpus = line[10:].strip()
                elif line.lower().startswith("@attribute"):
                    feat_name, typ = line[11:].strip().rsplit(maxsplit=1)
                    if feat_name.lower() == "name":
                        continue
                    self.feature_names.append(feat_name)
                    if typ.lower() not in ["real", "numeric", "integer"]:
                        raise ValueError(f"Incorrect attribute type for {feat_name}.")
                elif line.lower().startswith("@data"):
                    break
            self._header = fid.tell()
            reader = csv.reader(fid, skipinitialspace=True)
            _names = [row[0] for row in reader]
            counts: typing.Counter[str] = Counter(_names)
            self.names = list(counts.keys())
            self.slices = np.array(list(counts.values()))

    def read(self) -> Iterator[np.ndarray]:
        self.feature_names = []
        with open(self.path) as fid:
            fid.seek(self._header)
            reader = csv.reader(fid, skipinitialspace=True)
            # rows = itertools.islice(reader, self._header, None)  # Skip header
            for length in self.slices:
                yield np.array([x[1:] for x in itertools.islice(reader, length)])


class SequentialWriter:
    """Write features to file.

    Parameters
    ----------
    names : List[str]
        List of utterance names.
    corpus : str
        Corpus name.
    feature_names : List[str]
        List of feature names.
    """

    def __init__(
        self,
        names: List[str],
        corpus: str = "",
        feature_names: List[str] = [],
    ) -> None:
        self.names = names
        self.corpus = corpus
        if len(feature_names) == 0:
            raise ValueError("Feature names must be provided.")
        self.feature_names = feature_names

    def write(self, path: PathOrStr, features: Iterable[np.ndarray], **kwargs):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        meth = SequentialWriter._write_backends.get(path.suffix)
        if meth is None:
            raise ValueError(f"File format {path.suffix} not supported.")
        meth(self, path, features, **kwargs)

    def write_arff(self, path: PathOrStr, features: Iterable[np.ndarray], **kwargs):
        """Write features to ARFF file."""

        with open(path, "w") as fid:
            fid.write(f"@RELATION {self.corpus}\n\n")
            fid.write("@ATTRIBUTE name STRING\n")
            for feat in self.feature_names:
                fid.write(f"@ATTRIBUTE {feat} NUMERIC\n")
            fid.write("\n@data\n")
            writer = csv.writer(fid)
            for name, inst in zip(self.names, features):
                if len(inst.shape) == 2:
                    for row in inst:
                        writer.writerow([name] + list(row))
                else:
                    writer.writerow([name] + list(inst))

    def write_csv(self, path, features: Iterable[np.ndarray], header=True, **kwargs):
        """Write features to CSV file."""

        with open(path, "w") as fid:
            writer = csv.writer(fid, lineterminator="\n")
            if header:
                writer.writerow(["name"] + self.feature_names)
            for name, inst in zip(self.names, features):
                if len(inst.shape) == 2:
                    for row in inst:
                        writer.writerow([name] + list(row))
                else:
                    writer.writerow([name] + list(inst))

    def write_netcdf(
        self,
        path: PathOrStr,
        features: Iterable[np.ndarray],
        chunksize: int = 1024,
        **kwargs,
    ):
        """Write features to NetCDF4 file."""

        dataset = netCDF4.Dataset(path, "w")
        dataset.createDimension("name", len(self.names))
        dataset.createDimension("concat", 0)

        # Need to get first array for metadata
        features, _tmp = itertools.tee(features)
        _x0 = next(_tmp)
        del _tmp
        dim = _x0.shape[-1]
        dtype = _x0.dtype
        if dtype.kind == "f":
            # Force single-precision for consistency and space
            dtype = np.float32
        dataset.createDimension("feature", dim)

        filename = dataset.createVariable("name", str, ("name",))
        filename[:] = np.array(self.names)

        _features = dataset.createVariable(
            "data", dtype, ("concat", "feature"), chunksizes=(chunksize, dim)
        )
        slices = []
        idx = 0
        for x in features:
            i = 1 if x.ndim == 1 else len(x)
            _features[idx : idx + i, :] = x
            slices.append(i)
            idx += i

        _slices = dataset.createVariable("slices", int, ("name",))
        _slices[:] = np.array(slices)

        _feature_names = dataset.createVariable("feature", str, ("feature",))
        _feature_names[:] = np.array(self.feature_names)

        dataset.setncattr_string("corpus", self.corpus)
        dataset.setncattr_string("ertk_data_version", "2")
        dataset.close()

    def write_raw(
        self, path: PathOrStr, features: Iterable[np.ndarray], sr: int = 16000, **kwargs
    ):
        """Write raw audio files."""

        output_dir = Path(path).with_suffix("")
        output_dir.mkdir(exist_ok=True, parents=True)
        for name, audio in zip(self.names, features):
            output_path = output_dir / f"{name}.wav"
            soundfile.write(output_path, audio, subtype="PCM_16", samplerate=sr)
        write_filelist(output_dir.glob("*.wav"), path)

    _write_backends: Dict[str, Callable] = {
        ".arff": write_arff,
        ".csv": write_csv,
        ".nc": write_netcdf,
        ".txt": write_raw,
    }


_READ_BACKENDS_MEM: Dict[str, Callable[..., FeaturesData]] = {
    ".nc": read_netcdf_mem,
    ".arff": read_arff_mem,
    ".csv": read_csv_mem,
    ".txt": read_raw_mem,
}

_READ_BACKENDS_SEQUENTIAL: Dict[str, Type[SequentialReader]] = {
    ".nc": NetCDFReader,
    ".arff": ARFFSequentialReader,
    ".csv": CSVSequentialReader,
    ".txt": RawAudioReader,
}


def register_format(
    suffix: str,
    read: Optional[Callable[..., FeaturesData]],
    write: Optional[Callable[..., None]],
):
    """Register new read/write functions for a given file format."""

    if suffix[0] != ".":
        suffix = "." + suffix
    if read is not None:
        _READ_BACKENDS_MEM[suffix] = read
    if write is not None:
        SequentialWriter._write_backends[suffix] = write


def find_features_file(files: Iterable[Path]) -> Path:
    files = list(files)
    for suffix in _READ_BACKENDS_MEM:
        file = next((x for x in files if x.suffix == suffix), None)
        if file is not None:
            return file
    raise FileNotFoundError(f"No features found in {files}")


def read_features(path: PathOrStr, **kwargs) -> FeaturesData:
    """Read features from given path into memory."""

    path = Path(path)
    meth = _READ_BACKENDS_MEM.get(path.suffix)
    if meth is None:
        raise ValueError(f"File format {path.suffix} not supported.")
    logger.debug(f"Reading features from {path}")
    return meth(path, **filter_kwargs(kwargs, meth))


def read_features_iterable(path: PathOrStr, **kwargs) -> SequentialReader:
    """Read features from given path either sequentially or by index."""

    path = Path(path)
    meth = _READ_BACKENDS_SEQUENTIAL.get(path.suffix)
    if meth is None:
        raise ValueError(f"File format {path.suffix} not supported.")
    logger.debug(f"Reading features from {path}")
    return meth(path, **filter_kwargs(kwargs, meth.read))


def write_features(
    path: PathOrStr,
    features: Union[Iterable[np.ndarray], np.ndarray],
    names: List[str],
    corpus: str = "",
    feature_names: List[str] = [],
    slices: Union[np.ndarray, List[int], None] = None,
    **kwargs,
) -> None:
    """Convenience function to write features to given path.

    Parameters
    ----------
    path : PathOrStr
        Path to write features to.
    features : Union[Iterable[np.ndarray], np.ndarray]
        Features to write. If a 3D array is given, the dimensions are
        assumed to be (instances, frames, features). If a 2D array is
        given, the dimensions are assumed to be (frames, features), and
        the `slices` parameter is required. If an iterable of arrays is
        given, each 1D or 2D array is assumed to be a single instance.
    names : List[str]
        Instance names.
    corpus : str
        Name of corpus. Default is empty string.
    feature_names : List[str]
        Feature names.
    slices : Union[np.ndarray, List[int]], optional
        Number of frames per instance. Only required if a 2D matrix is
        given, and the first dimension is different than the number of
        instances.
    """
    if slices is not None:
        if not isinstance(features, np.ndarray):
            raise ValueError("`slices` should only be given for a 2D contiguous array.")
        features = flat_to_inst(features, slices)

    SequentialWriter(names=names, corpus=corpus, feature_names=feature_names).write(
        path, features, **kwargs
    )
