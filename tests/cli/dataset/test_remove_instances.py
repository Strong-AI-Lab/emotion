from pathlib import Path

import pytest

from ertk.cli.dataset.remove_instances import main


def test_noargs():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_badargs():
    with pytest.raises(SystemExit) as excinfo:
        main(["x", "y"])
    assert excinfo.value.code == 2


test_data_dir = Path(__file__).parent / "../../test_data"


def test_remove_instances(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                f"{test_data_dir}/features/features_2d.csv",
                "--names",
                f"{test_data_dir}/subset.txt",
                f"{tmp_path}/removed.csv",
            ]
        )
    assert excinfo.value.code == 0
