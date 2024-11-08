from pathlib import Path
from typing import Optional

import arff
import netCDF4
import numpy as np
import pandas as pd
import pytest

from ertk.dataset.features import FeaturesData, read_features, write_features

from .constants import (
    all_clips_names,
    corpus_name,
    feature_names,
    features_2d,
    features_3d,
    features_dir,
    features_vlen,
    resample_dir,
    slices_vlen,
)


@pytest.mark.parametrize("format", [".nc", ".csv", ".arff"])
def test_read_features_2d(format: str):
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
def test_read_features_3d(format: str):
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
def test_read_features_vlen(format: str):
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
    slices: Optional[list[int]],
    tmp_path: Path,
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
            data = arff.load(fid)
        with open(ref_path) as fid:
            ref_data = arff.load(fid)
        assert data == ref_data
    elif format == ".nc":
        with netCDF4.Dataset(out_path) as data, netCDF4.Dataset(ref_path) as ref_data:
            assert np.array_equal(data.variables["data"], ref_data.variables["data"])
            assert np.array_equal(data.variables["name"], ref_data.variables["name"])
            assert np.array_equal(
                data.variables["feature"], ref_data.variables["feature"]
            )
            assert np.array_equal(
                data.variables["slices"], ref_data.variables["slices"]
            )
            assert data.corpus == ref_data.corpus


def test_featuresdata_flat():
    data = FeaturesData(np.array([[0.1, 0.2], [0.2, 0.1]]), ["test1", "test2"])
    assert hasattr(data, "flat")


@pytest.mark.parametrize("names", [["test1", "test2"], ["name2", "name1"]])
def test_featuresdata_names(names: list[str]):
    data = FeaturesData(np.array([[0.1, 0.2], [0.2, 0.1]]), names)
    assert data.names == names
    assert data.names is not names


@pytest.mark.parametrize("feature_names", [["f", "g"], ["f1", "f2"], None])
def test_featuresdata_feature_names(feature_names: Optional[list[str]]):
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


def test_featuresdata_2d():
    data = FeaturesData(features_2d, all_clips_names)
    data.make_contiguous()
    assert len(data.flat.shape) == 2
    assert np.array_equal(data.features, data.flat)
    assert len(data.features) == len(all_clips_names)
    assert len(data.slices) == len(all_clips_names)
    assert list(data.slices) == [1] * len(all_clips_names)


def test_featuresdata_3d():
    n_features = features_3d.shape[2]
    data = FeaturesData(features_3d, all_clips_names)
    data.make_contiguous()
    assert len(data.flat.shape) == 2
    assert len(data.features.shape) == 3
    assert np.array_equal(data.features.reshape(-1, n_features), data.flat)
    assert len(data.features) == len(all_clips_names)
    assert list(data.slices) == [features_3d.shape[1]] * len(all_clips_names)


def test_featuresdata_vlen():
    data = FeaturesData(features_vlen, all_clips_names, slices=slices_vlen)
    data.make_contiguous()
    assert len(data.flat.shape) == 2
    assert data.features.dtype == object
    assert len(data.features) == len(all_clips_names)
    assert list(data.slices) == slices_vlen
