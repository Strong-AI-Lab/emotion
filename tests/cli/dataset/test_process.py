from pathlib import Path

import pytest

from ertk.cli.dataset.process import main


def test_noargs():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_badargs():
    with pytest.raises(SystemExit) as excinfo:
        main(["x", "y"])
    assert excinfo.value.code == 2


test_data_dir = Path(__file__).parent / "../../test_data"


def test_process_opensmile(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--processor",
                "opensmile",
                f"{test_data_dir}/all_clips.txt",
                f"{tmp_path}/opensmile.csv",
                "opensmile_config=eGeMAPS",
            ]
        )
    assert excinfo.value.code == 0
