import csv
import itertools
import typing
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

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


def read_arff(path: PathOrStr, label: bool = False):
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


def read_csv(path: PathOrStr, header: bool = True, label: bool = False):
    df = pd.read_csv(path, header=0 if header else None, converters={0: str})
    counts: typing.Counter[str] = Counter(df.iloc[:, 0])
    idx = slice(1, -1) if label else slice(1, None)

    return FeaturesData(
        names=list(counts.keys()),
        features=np.array(df.iloc[:, idx]),
        slices=np.array(list(counts.values())),
        feature_names=list(df.columns[idx]),
    )


def read_netcdf(path: PathOrStr):
    with netCDF4.Dataset(path) as dataset:
        return FeaturesData(
            corpus=dataset.corpus,
            names=list(dataset.variables["name"]),
            features=np.array(dataset.variables["features"], copy=False),
            slices=np.array(dataset.variables["slices"]),
            feature_names=list(dataset.variables["feature_names"]),
        )


def read_raw(path: PathOrStr, sample_rate: int = 16000):
    path = Path(path)
    filepaths = get_audio_paths(path)
    _audio = []

    def read_one_file(filepath):
        with warnings.catch_warnings():
            audio, _ = librosa.load(
                filepath, sr=sample_rate, mono=True, res_type="kaiser_fast"
            )
        return np.expand_dims(audio, -1)

    _audio = TqdmParallel(len(filepaths), desc="Reading audio", leave=False, n_jobs=-1)(
        delayed(read_one_file)(filepath) for filepath in filepaths
    )

    return FeaturesData(
        corpus=path.parent.stem,
        names=[x.stem for x in filepaths],
        features=np.concatenate(_audio),
        slices=[len(x) for x in _audio],
        feature_names=["pcm"],
    )


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
            self._slices = np.ones(len(features), dtype=int)
            if len(features.shape) != 2 and features[0].ndim > 1:
                self._slices = np.array([len(x) for x in features])

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

    def write(self, path: PathOrStr, **kwargs):
        """Write features to any file format."""

        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        meth = FeaturesData._write_backends.get(path.suffix)
        if meth is None:
            raise ValueError(f"File format {path.suffix} not supported.")
        meth(self, path, **filter_kwargs(kwargs, meth))

    def write_arff(self, path: PathOrStr):
        data: Dict[str, Any] = {"relation": self.corpus}
        data["attributes"] = [("name", "STRING")] + [
            (x, "NUMERIC") for x in self.feature_names
        ]
        data["data"] = []
        # The liac-arff library can only dump all at once
        self.make_contiguous()
        for name, row in zip(np.repeat(self.names, self.slices), self.flat):
            data["data"].append([name] + list(row))

        with open(path, "w") as fid:
            arff.dump(data, fid)

    def write_csv(self, path, header: bool = True):
        with open(path, "w") as fid:
            writer = csv.writer(fid)
            if header:
                writer.writerow(["name"] + self.feature_names)
            for name, inst in zip(self.names, self.features):
                if len(inst.shape) == 2:
                    for row in inst:
                        writer.writerow([name] + list(row))
                else:
                    writer.writerow([name] + list(inst))

    def write_netcdf(self, path: PathOrStr):
        dataset = netCDF4.Dataset(path, "w")
        dataset.createDimension("instance", len(self.names))
        dataset.createDimension("concat", sum(self.slices))
        dataset.createDimension("features", self.features[0].shape[-1])

        _slices = dataset.createVariable("slices", int, ("instance",))
        _slices[:] = np.array(self.slices)

        filename = dataset.createVariable("name", str, ("instance",))
        filename[:] = np.array(self.names)

        _features = dataset.createVariable(
            "features", np.float32, ("concat", "features")
        )
        idx = np.cumsum(self.slices)
        for i, x in enumerate(self.features):
            end = idx[i]
            start = end - self.slices[i]
            _features[start:end, :] = x

        _feature_names = dataset.createVariable("feature_names", str, ("features",))
        _feature_names[:] = np.array(self.feature_names)

        dataset.setncattr_string("corpus", self.corpus)
        dataset.close()

    def write_raw(self, path: PathOrStr, sr: int = 16000):
        output_dir = Path(path).with_suffix("")
        output_dir.mkdir(exist_ok=True, parents=True)
        for name, audio in zip(self.names, self.features):
            output_path = output_dir / f"{name}.wav"
            soundfile.write(output_path, audio, subtype="PCM_16", samplerate=sr)
        write_filelist(output_dir.glob("*.wav"), path)

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

    _write_backends: Dict[str, Callable] = {
        ".arff": write_arff,
        ".csv": write_csv,
        ".nc": write_netcdf,
        ".txt": write_raw,
    }


class IterableReader(ABC):
    feature_names: List[str]
    names: List[str]
    corpus: str = ""

    def __init__(self, path: PathOrStr, **kwargs) -> None:
        self.path = path
        self.read_metadata()
        self.iter = iter(self.read(**kwargs))

    def __iter__(self):
        return self.iter

    def __len__(self):
        return len(self.names)

    @abstractmethod
    def read_metadata(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def read(self, **kwargs) -> Iterable[np.ndarray]:
        raise NotImplementedError()


class CSVIterableReader(IterableReader):
    def read_metadata(self) -> None:
        with open(self.path) as fid:
            reader = csv.reader(fid, newline="")
            _, *feature_names = next(reader)
            self.feature_names = feature_names
            _names = [row[0] for row in reader]
            counts: typing.Counter[str] = Counter(_names)
            self.names = list(counts.keys())
            self.slices = list(counts.values())

    def read(self, **kwargs) -> Iterable[np.ndarray]:
        with open(self.path) as fid:
            reader = csv.reader(fid, newline="")
            next(reader)  # Skip header
            for length in self.slices:
                yield np.array([x[1:] for x in itertools.islice(reader, length)])


class ARFFIterableReader(IterableReader):
    def read_metadata(self) -> None:
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
            reader = csv.reader(fid, skipinitialspace=True, newline="")
            _names = [row[0] for row in reader]
            counts: typing.Counter[str] = Counter(_names)
            self.names = list(counts.keys())
            self.slices = list(counts.values())

    def read(self, **kwargs) -> Iterable[np.ndarray]:
        self.feature_names = []
        with open(self.path) as fid:
            fid.seek(self._header)
            reader = csv.reader(fid, skipinitialspace=True, newline="")
            # rows = itertools.islice(reader, self._header, None)  # Skip header
            for length in self.slices:
                yield np.array([x[1:] for x in itertools.islice(reader, length)])


class NetCDFIterableReader(IterableReader):
    def read_metadata(self) -> None:
        with netCDF4.Dataset(self.path) as dataset:
            self.corpus = dataset.corpus
            self.names = list(dataset.variables["name"])
            self.feature_names = list(dataset.variables["feature_names"])
            self.slices = np.array(dataset.variables["slices"])

    def read(self, **kwargs) -> Iterable[np.ndarray]:
        with netCDF4.Dataset(self.path) as dataset:
            idx = np.r_[0, np.cumsum(self.slices)]
            for start, end in zip(idx[:-1], idx[1:]):
                yield np.array(dataset.variables["features"][start:end])


class RawAudioIterableReader(IterableReader):
    def read_metadata(self) -> None:
        self.filepaths = get_audio_paths(self.path)
        self.names = [x.stem for x in self.filepaths]
        self.feature_names = ["pcm"]

    def read(self, sample_rate: int = 16000, **kwargs) -> Iterable[np.ndarray]:
        with warnings.catch_warnings():
            for filepath in self.filepaths:
                audio, _ = librosa.load(
                    filepath, sr=sample_rate, mono=True, res_type="kaiser_fast"
                )
                yield np.expand_dims(audio, -1)


class IterableWriter:
    def __init__(
        self,
        names: List[str],
        corpus: str = "",
        feature_names: List[str] = [],
    ) -> None:
        self.names = names
        self.corpus = corpus
        self.feature_names = feature_names

    def write(self, path: PathOrStr, features: Iterable[np.ndarray], **kwargs):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        meth = IterableWriter._write_backends.get(path.suffix)
        if meth is None:
            raise ValueError(f"File format {path.suffix} not supported.")
        meth(self, path, features, **kwargs)

    def write_arff(self, path: PathOrStr, features: Iterable[np.ndarray], **kwargs):
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

    def write_csv(self, path, features: Iterable[np.ndarray], **kwargs):
        with open(path, "w") as fid:
            writer = csv.writer(fid)
            writer.writerow(["name"] + self.feature_names)
            for name, inst in zip(self.names, features):
                if len(inst.shape) == 2:
                    for row in inst:
                        writer.writerow([name] + list(row))
                else:
                    writer.writerow([name] + list(inst))

    def write_netcdf(self, path: PathOrStr, features: Iterable[np.ndarray], **kwargs):
        dataset = netCDF4.Dataset(path, "w")
        dataset.createDimension("instance", len(self.names))
        dataset.createDimension("concat", 0)

        features, _tmp = itertools.tee(features)
        _x0 = next(_tmp)
        del _tmp
        dataset.createDimension("features", _x0.shape[-1])

        filename = dataset.createVariable("name", str, ("instance",))
        filename[:] = np.array(self.names)

        _features = dataset.createVariable(
            "features", np.float32, ("concat", "features")
        )
        slices = []
        idx = 0
        for x in features:
            _features[idx : idx + len(x), :] = x
            slices.append(len(x))
            idx += len(x)

        _slices = dataset.createVariable("slices", int, ("instance",))
        _slices[:] = np.array(slices)

        _feature_names = dataset.createVariable("feature_names", str, ("features",))
        _feature_names[:] = np.array(self.feature_names)

        dataset.setncattr_string("corpus", self.corpus)
        dataset.close()

    def write_raw(
        self, path: PathOrStr, features: Iterable[np.ndarray], sr: int = 16000, **kwargs
    ):
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


_READ_BACKENDS: Dict[str, Callable[..., FeaturesData]] = {
    ".nc": read_netcdf,
    ".arff": read_arff,
    ".csv": read_csv,
    ".txt": read_raw,
}

_READ_BACKENDS_ITERABLE: Dict[str, Type[IterableReader]] = {
    ".nc": NetCDFIterableReader,
    ".arff": ARFFIterableReader,
    ".csv": CSVIterableReader,
    ".txt": RawAudioIterableReader,
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
        _READ_BACKENDS[suffix] = read
    if write is not None:
        FeaturesData._write_backends[suffix] = write
        IterableWriter._write_backends[suffix] = write


def find_features_file(files: Iterable[Path]) -> Path:
    files = list(files)
    for suffix in _READ_BACKENDS:
        file = next((x for x in files if x.suffix == suffix), None)
        if file is not None:
            return file
    raise FileNotFoundError(f"No features found in {files}")


def read_features(path: PathOrStr, **kwargs) -> FeaturesData:
    """Read features from given path."""

    path = Path(path)
    meth = _READ_BACKENDS.get(path.suffix)
    if meth is None:
        raise ValueError(f"File format {path.suffix} not supported.")
    return meth(path, **filter_kwargs(kwargs, meth))


def read_features_iterable(path: PathOrStr, **kwargs) -> IterableReader:
    """Read features from given path as an iterable."""

    path = Path(path)
    meth = _READ_BACKENDS_ITERABLE.get(path.suffix)
    if meth is None:
        raise ValueError(f"File format {path.suffix} not supported.")
    return meth(path, **filter_kwargs(kwargs, meth.read))


def write_features(
    path: PathOrStr,
    features: Union[Iterable[np.ndarray], np.ndarray],
    names: List[str],
    corpus: str = "",
    feature_names: List[str] = [],
    slices: Union[np.ndarray, List[int]] = None,
    **kwargs,
) -> None:
    """Convenience function to write features to given path. Arguments
    are the same as those passed to IterableWriter.
    """
    if slices is not None:
        if not isinstance(features, np.ndarray):
            raise ValueError("`slices` should only be given for a 2D contiguous array.")
        features = flat_to_inst(features, slices)

    IterableWriter(names=names, corpus=corpus, feature_names=feature_names).write(
        path, features
    )
