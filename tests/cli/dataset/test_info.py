from pathlib import Path

import pytest

from ertk.cli.dataset.info import main


def test_noargs():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_badargs():
    with pytest.raises(SystemExit) as excinfo:
        main(["x", "y"])
    assert excinfo.value.code == 2


test_data_dir = Path(__file__).parent / "../../test_data"


def test_info():
    with pytest.raises(SystemExit) as excinfo:
        main([f"{test_data_dir}/corpus_info.yaml"])
    assert excinfo.value.code == 0


def test_info_subset():
    with pytest.raises(SystemExit) as excinfo:
        main([f"{test_data_dir}/corpus_info.yaml", "--subset", "subset1"])
    assert excinfo.value.code == 0
