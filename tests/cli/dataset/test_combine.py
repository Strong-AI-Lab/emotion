from pathlib import Path

import pytest

from ertk.cli.dataset.combine import main


def test_noargs():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_badargs():
    with pytest.raises(SystemExit) as excinfo:
        main(["x", "y"])
    assert excinfo.value.code == 2


test_data_dir = Path(__file__).parent / "../../test_data"


def test_combine_same(tmp_path):
    with pytest.raises(ValueError):
        main(
            [
                "--prefix_corpus",
                f"{test_data_dir}/features/features_2d.csv",
                f"{test_data_dir}/features/features_2d.csv",
                f"{tmp_path}/combined.csv",
            ]
        )


def test_combine_single(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                f"{test_data_dir}/features/features_2d.csv",
                f"{tmp_path}/combined.csv",
            ]
        )
    assert excinfo.value.code == 0
