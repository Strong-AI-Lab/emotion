from pathlib import Path

import pytest

from ertk.cli.dataset.combine import main


def test_noargs():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_badargs():
    with pytest.raises(SystemExit) as excinfo:
        main(["x"])
    assert excinfo.value.code == 2


test_data_dir = Path(__file__).parent / "../../test_data"


def test_combine_same(tmp_path):
    with pytest.raises(ValueError, match="not unique"):
        main(
            [
                f"{test_data_dir}/features/features_2d.csv",
                f"{test_data_dir}/features/features_2d.csv",
                f"{tmp_path}/combined.csv",
            ]
        )


def test_combine_csv(tmp_path):
    with pytest.raises(ValueError, match="corpus name"):
        main(
            [
                "--prefix_corpus",
                f"{test_data_dir}/features/features_2d.csv",
                f"{tmp_path}/combined.csv",
            ]
        )


def test_combine_size(tmp_path):
    with pytest.raises(ValueError, match="Feature size"):
        main(
            [
                f"{test_data_dir}/features/features_2d.csv",
                f"{test_data_dir}/features/features_2d_small.csv",
                f"{tmp_path}/combined.csv",
            ]
        )


def test_combine_same_prefix(tmp_path):
    with pytest.raises(ValueError, match="not unique"):
        main(
            [
                "--prefix_corpus",
                f"{test_data_dir}/features/features_2d.nc",
                f"{test_data_dir}/features/features_2d.nc",
                f"{tmp_path}/combined.nc",
            ]
        )


def test_combine_prefix_corpus(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--prefix_corpus",
                f"{test_data_dir}/features/features_2d_split1.nc",
                f"{test_data_dir}/features/features_2d_split2.nc",
                f"{tmp_path}/combined.nc",
            ]
        )
    assert excinfo.value.code == 0


def test_combine_single(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                f"{test_data_dir}/features/features_2d.csv",
                f"{tmp_path}/combined.csv",
            ]
        )
    assert excinfo.value.code == 0
