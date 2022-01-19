from pathlib import Path
from typing import List, Optional

import arff
import librosa
import netCDF4
import numpy as np
import pandas as pd
import pytest
from constants import (
    all_clips_names,
    all_clips_unsorted_names,
    corpus_name,
    feature_names,
    features_2d,
    features_3d,
    features_vlen,
    slices_vlen,
    subset_names,
)

from ertk.dataset.annotation import read_annotations, write_annotations
from ertk.dataset.features import FeaturesData, read_features, write_features
from ertk.dataset.utils import (
    get_audio_paths,
    resample_audio,
    resample_rename_clips,
    write_filelist,
)


#
# Fixtures
#
@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parent / "test_data"


@pytest.fixture
def audio_dir(test_data_dir: Path) -> Path:
    return test_data_dir / "data"


@pytest.fixture
def resample_dir(test_data_dir: Path) -> Path:
    return test_data_dir / "resampled"


@pytest.fixture
def features_dir(test_data_dir: Path) -> Path:
    return test_data_dir / "features"


@pytest.fixture
def resample_paths(audio_dir: Path) -> List[Path]:
    return list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.flv"))


#
# Utils
#
def test_get_audio_paths_dir(audio_dir: Path) -> None:
    paths = get_audio_paths(audio_dir)
    assert len(paths) == 12


def test_get_audio_paths_file_1(test_data_dir: Path) -> None:
    paths = get_audio_paths(test_data_dir / "all_clips.txt")
    assert len(paths) == 12


def test_get_audio_paths_file_2(test_data_dir: Path) -> None:
    paths = get_audio_paths(test_data_dir / "subset.txt")
    assert len(paths) == 6


# This one is redundant since test_data_dir is absolute due to __file__
def test_get_audio_paths_absolute(audio_dir: Path) -> None:
    paths = get_audio_paths(audio_dir, absolute=True)
    for path in paths:
        assert path.is_absolute()
        assert path == audio_dir / path.name
    assert len(paths) == 12


def test_resample(
    tmp_path: Path, resample_paths: List[Path], resample_dir: Path
) -> None:
    resample_audio(resample_paths, tmp_path)
    paths = list(tmp_path.glob("*.wav"))
    for path in paths:
        audio, sr = librosa.load(path, sr=None)
        assert sr == 16000
        ref, _ = librosa.load(resample_dir / path.name, sr=None)
        assert np.allclose(audio, ref)
    assert len(paths) == 12


def test_resample_rename(tmp_path: Path, resample_paths: List[Path]) -> None:
    mapping = {x: tmp_path / (x.stem + "_rename.wav") for x in resample_paths}
    resample_rename_clips(mapping)
    paths = list(tmp_path.glob("*.wav"))
    for path in paths:
        assert path.name.endswith("_rename.wav")
        _, sr = librosa.load(path, sr=None, duration=0)
        assert sr == 16000
    assert len(paths) == 12


def test_write_filelist_name() -> None:
    paths = ["test_clip1.wav", "clip2.mp3", "clip3.ogg"]
    write_filelist([Path(x) for x in paths], "test_filelist")
    with open("test_filelist.txt") as fid:
        lines = list(map(str.strip, fid))
    assert lines == sorted(paths)


def test_write_filelist_path(tmp_path: Path) -> None:
    paths = ["test_clip1.wav", "clip2.mp3", "clip3.ogg"]
    out_path = tmp_path / "test_filelist.txt"
    write_filelist([Path(x) for x in paths], out_path)
    with open(out_path) as fid:
        lines = list(map(str.strip, fid))
    assert lines == sorted(paths)


#
# Annotations
#
def test_read_annotations_str_1(test_data_dir: Path) -> None:
    annotations = read_annotations(test_data_dir / "annot2_str.csv", dtype=str)
    assert type(annotations["extra1"]) is str
    assert annotations["extra1"] == "1"
    assert annotations["extra2"] == "0"
    assert annotations["1002_DFA_DIS_XX"] == "1"
    assert annotations["1002_DFA_HAP_XX"] == "3"


def test_read_annotations_str_2(test_data_dir: Path) -> None:
    annotations = read_annotations(test_data_dir / "label.csv", dtype=str)
    assert type(annotations["extra1"]) is str
    assert annotations["extra1"] == "none"
    assert annotations["extra2"] == "sadness"
    assert annotations["1002_DFA_DIS_XX"] == "disgust"
    assert annotations["1002_DFA_HAP_XX"] == "happiness"


def test_read_annotations_float(test_data_dir: Path) -> None:
    annotations = read_annotations(test_data_dir / "annot1.csv", dtype=float)
    assert type(annotations["extra1"]) is float
    assert annotations["extra1"] == 0
    assert annotations["extra2"] == 1
    assert annotations["1001_DFA_HAP_XX"] == pytest.approx(3.8472069)
    assert annotations["1001_DFA_NEU_XX"] == pytest.approx(6.9999)


def test_write_annotations_str(tmp_path: Path) -> None:
    mapping = {"clip1": "x", "clip2": "y", "clip3": "z"}
    out_path = tmp_path / "annotaton.csv"
    write_annotations(mapping, "test_annot", out_path)
    df = pd.read_csv(out_path, header=0, index_col=0, dtype=str)
    assert list(df.index) == ["clip1", "clip2", "clip3"]
    assert df.columns[0] == "test_annot"
    assert df.loc["clip1", "test_annot"] == "x"


def test_write_annotations_float(tmp_path: Path) -> None:
    mapping = {"clip1": 1.5, "clip2": 2, "clip3": 0.9}
    out_path = tmp_path / "annotaton_float.csv"
    write_annotations(mapping, "test_annot2", out_path)
    df = pd.read_csv(out_path, header=0, index_col=0, dtype={0: str, 1: float})
    assert list(df.index) == ["clip1", "clip2", "clip3"]
    assert df.columns[0] == "test_annot2"
    assert df.loc["clip2", "test_annot2"] == 2
    assert df.loc["clip3", "test_annot2"] == pytest.approx(0.9)


#
# Features
#
@pytest.mark.parametrize("format", [".nc", ".csv", ".arff"])
def test_read_features_2d(format: str, features_dir: Path, resample_dir: Path) -> None:
    features_path = features_dir / ("features_2d" + format)
    data = read_features(features_path)
    if format != ".csv":
        assert data.corpus == "test_corpus"
    assert data.feature_names == [f"test_feat{i}" for i in range(15)]
    assert data.names == sorted(x.stem for x in resample_dir.glob("*"))
    assert data.flat.shape == (12, 15)
    assert data.flat.flags.forc
    assert data.features.shape == (12, 15)
    assert data.features is data.flat
    assert list(data.slices) == [1] * 12
    assert np.allclose(data.features, features_2d)


@pytest.mark.parametrize("format", [".nc", ".csv", ".arff"])
def test_read_features_3d(format: str, features_dir: Path, resample_dir: Path) -> None:
    features_path = features_dir / ("features_3d" + format)
    data = read_features(features_path)
    if format != ".csv":
        assert data.corpus == "test_corpus"
    assert data.feature_names == [f"test_feat{i}" for i in range(15)]
    assert data.names == sorted(x.stem for x in resample_dir.glob("*"))
    assert data.flat.shape == (120, 15)
    assert data.flat.flags.forc
    assert data.features.shape == (12, 10, 15)
    assert list(data.slices) == [10] * 12
    assert np.allclose(data.features, features_3d)


@pytest.mark.parametrize("format", [".nc", ".csv", ".arff"])
def test_read_features_vlen(
    format: str, features_dir: Path, resample_dir: Path
) -> None:
    features_path = features_dir / ("features_vlen" + format)
    data = read_features(features_path)
    if format != ".csv":
        assert data.corpus == "test_corpus"
    assert data.feature_names == [f"test_feat{i}" for i in range(15)]
    assert data.names == sorted(x.stem for x in resample_dir.glob("*"))
    assert data.flat.shape == (60, 15)
    assert data.flat.flags.forc
    assert data.features.shape == (12,)
    assert list(data.slices) == [3, 2, 9, 9, 4, 4, 7, 4, 6, 2, 7, 3]
    assert np.allclose(data.flat, features_vlen)


@pytest.mark.parametrize(
    ["name", "features", "slices"],
    [
        ("features_2d", features_2d, None),
        ("features_3d", features_3d, None),
        ("features_vlen", features_vlen, slices_vlen),
    ],
)
@pytest.mark.parametrize("format", [".nc", ".csv", ".arff"])
def test_write_features(
    format: str,
    name: str,
    features: np.ndarray,
    slices: Optional[List[int]],
    tmp_path: Path,
    features_dir: Path,
):
    filename = name + format
    ref_path = features_dir / filename
    out_path = tmp_path / filename
    write_features(
        out_path,
        features,
        all_clips_names,
        corpus=corpus_name,
        feature_names=feature_names,
        slices=slices,
    )
    if format == ".csv":
        df1 = pd.read_csv(out_path, header=0, index_col=0)
        df2 = pd.read_csv(ref_path, header=0, index_col=0)
        assert df1.equals(df2)
    elif format == ".arff":
        with open(out_path) as fid:
            data1 = arff.load(fid)
        with open(ref_path) as fid:
            data2 = arff.load(fid)
        assert data1 == data2
    elif format == ".nc":
        with netCDF4.Dataset(out_path) as data1, netCDF4.Dataset(ref_path) as data2:
            assert np.array_equal(
                data1.variables["features"], data2.variables["features"]
            )
            assert np.array_equal(data1.variables["name"], data2.variables["name"])
            assert np.array_equal(
                data1.variables["feature_names"], data2.variables["feature_names"]
            )
            assert np.array_equal(data1.variables["slices"], data2.variables["slices"])
            assert data1.corpus == data2.corpus


def test_featuresdata_flat() -> None:
    data = FeaturesData(np.array([[0.1, 0.2], [0.2, 0.1]]), ["test1", "test2"])
    with pytest.raises(AttributeError):
        _ = data.flat


@pytest.mark.parametrize("names", [["test1", "test2"], ["name2", "name1"]])
def test_featuresdata_names(names: List[str]) -> None:
    data = FeaturesData(np.array([[0.1, 0.2], [0.2, 0.1]]), names)
    assert data.names == names
    assert data.names is not names


@pytest.mark.parametrize("feature_names", [["f", "g"], ["f1", "f2"], None])
def test_featuresdata_feature_names(feature_names: Optional[List[str]]) -> None:
    data = FeaturesData(
        np.array([[0.1, 0.2], [0.2, 0.1]]),
        names=["test1", "test2"],
        feature_names=feature_names,
    )
    if feature_names is not None:
        assert data.feature_names == feature_names
    else:
        assert data.feature_names == [f"feature{i + 1}" for i in range(2)]
    assert data.feature_names is not feature_names


def test_featuresdata_2d() -> None:
    data = FeaturesData(features_2d, all_clips_names)
    data.make_contiguous()
    assert len(data.flat.shape) == 2
    assert np.array_equal(data.features, data.flat)
    assert len(data.features) == len(all_clips_names)
    assert len(data.slices) == len(all_clips_names)
    assert list(data.slices) == [1] * len(all_clips_names)


def test_featuresdata_3d() -> None:
    n_features = features_3d.shape[2]
    data = FeaturesData(features_3d, all_clips_names)
    data.make_contiguous()
    assert len(data.flat.shape) == 2
    assert len(data.features.shape) == 3
    assert np.array_equal(data.features.reshape(-1, n_features), data.flat)
    assert len(data.features) == len(all_clips_names)
    assert list(data.slices) == [features_3d.shape[1]] * len(all_clips_names)


def test_featuresdata_vlen() -> None:
    data = FeaturesData(features_vlen, all_clips_names, slices=slices_vlen)
    data.make_contiguous()
    assert len(data.flat.shape) == 2
    assert data.features.dtype == object
    assert len(data.features) == len(all_clips_names)
    assert list(data.slices) == slices_vlen
