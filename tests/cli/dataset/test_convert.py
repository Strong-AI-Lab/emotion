from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

from ertk.cli.dataset.convert import main


def test_noargs():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_badargs():
    with pytest.raises(SystemExit) as excinfo:
        main(["x", "y"])
    assert excinfo.value.code == 2


test_data_dir = Path(__file__).parent / "../../test_data"


def test_convert(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                f"{test_data_dir}/features/features_2d.nc",
                f"{tmp_path}/converted.csv",
            ]
        )
    assert excinfo.value.code == 0
    df_converted = pd.read_csv(f"{tmp_path}/converted.csv", index_col=0)
    df_original = pd.read_csv(f"{test_data_dir}/features/features_2d.csv", index_col=0)
    assert df_converted.columns.equals(df_original.columns)
    assert np.allclose(df_converted.values, df_original.values, equal_nan=True)


def test_convert_suffix(tmp_path):
    with pytest.raises(ValueError, match="must be different"):
        main(
            [
                f"{test_data_dir}/features/features_2d.nc",
                f"{tmp_path}/converted.nc",
            ]
        )


def test_convert_corpus(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                f"{test_data_dir}/features//features_2d.csv",
                f"{tmp_path}/converted.nc",
                "--corpus",
                "test",
            ]
        )
    assert excinfo.value.code == 0
    dataset = netCDF4.Dataset(f"{tmp_path}/converted.nc")
    assert dataset.corpus == "test"
