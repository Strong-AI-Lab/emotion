from pathlib import Path

import pytest

from ertk.cli.dataset.annotation_stats import main


def test_noargs():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_badargs():
    with pytest.raises(SystemExit) as excinfo:
        main(["x", "y"])
    assert excinfo.value.code == 2


test_data_dir = Path(__file__).parent / "../../test_data"


def test_annotation_stats():
    with pytest.raises(SystemExit) as excinfo:
        main([f"{test_data_dir}/label.csv"])
    assert excinfo.value.code == 0
