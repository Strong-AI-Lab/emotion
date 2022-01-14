import typing
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import arff
import librosa
import netCDF4
import numpy as np
import pandas as pd
import soundfile
from joblib import delayed

from ..utils import (
    PathOrStr,
    TqdmParallel,
    _make_array_array,
    filter_kwargs,
    flat_to_inst,
    inst_to_flat,
)
from .utils import get_audio_paths


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
        feature_names=df.columns[idx],
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

    Args:
    -----
    features: ndarray
        Features matrix. May be "flat" or per-instance. If flat,
        `slices` must also be present.
    names: list of str
        List of instance names.
    corpus: str
        Corpus for this dataset.
    feature_names: list of str
        List of feature names.
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
        feature_names: List[str] = None,
        slices: Union[np.ndarray, List[int], None] = None,
    ):
        self._corpus = corpus
        self._names = names
        if isinstance(features, list):
            features = _make_array_array(features)
        self._features = features
        self._feature_names = feature_names or []

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
        for name, row in zip(np.repeat(self.names, self.slices), self.flat):
            data["data"].append([name] + list(row))

        with open(path, "w") as fid:
            arff.dump(data, fid)

    def write_csv(self, path, header: bool = True):
        df = pd.DataFrame(self.flat, columns=self.feature_names)
        df.index = pd.Index(np.repeat(self.names, self.slices), name="name")
        df.to_csv(path, header=header)

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

    def write_raw(self, path: PathOrStr):
        output_dir = Path(path) / "audio"
        output_dir.mkdir(exist_ok=True, parents=True)
        for name, audio in zip(self.names, self.features):
            output_path = output_dir / (name + ".wav")
            soundfile.write(output_path, audio, subtype="PCM_16", samplerate=16000)

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


_READ_BACKENDS: Dict[str, Callable[..., FeaturesData]] = {
    ".nc": read_netcdf,
    ".arff": read_arff,
    ".csv": read_csv,
    ".txt": read_raw,
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


def write_features(
    path: PathOrStr,
    features: Union[List[np.ndarray], np.ndarray],
    names: List[str],
    corpus: str = "",
    feature_names: List[str] = None,
    slices: Union[np.ndarray, List[int]] = None,
    **kwargs,
):
    """Convenience function to write features to given path. Arguments
    are the same as those passed to FeaturesData.
    """

    FeaturesData(
        features, names, corpus=corpus, slices=slices, feature_names=feature_names
    ).write(path, **kwargs)
