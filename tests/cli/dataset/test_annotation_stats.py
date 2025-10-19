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


def test_annotation_stats_multiple():
    with pytest.raises(SystemExit) as excinfo:
        main([f"{test_data_dir}/label.csv", f"{test_data_dir}/speaker.csv"])
    assert excinfo.value.code == 0

    with pytest.raises(SystemExit) as excinfo:
        main([f"{test_data_dir}/label.csv", f"{test_data_dir}/annot1.csv"])
    assert excinfo.value.code == 0

    with pytest.raises(SystemExit) as excinfo:
        main([f"{test_data_dir}/annot1.csv", f"{test_data_dir}/label.csv"])
    assert excinfo.value.code == 0


def test_annotation_stats_dtype():
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                f"{test_data_dir}/label.csv",
                f"{test_data_dir}/speaker.csv",
                f"{test_data_dir}/annot2_str.csv",
                "--dtype",
                "str",
            ]
        )
    assert excinfo.value.code == 0


@pytest.mark.filterwarnings("ignore:non-interactive")
@pytest.mark.usefixtures("mpl")
def test_annotation_stats_plot():
    with pytest.raises(SystemExit) as excinfo:
        main([f"{test_data_dir}/label.csv", f"{test_data_dir}/speaker.csv", "--plot"])
    assert excinfo.value.code == 0

    with pytest.raises(SystemExit) as excinfo:
        main([f"{test_data_dir}/label.csv", f"{test_data_dir}/annot1.csv", "--plot"])
    assert excinfo.value.code == 0

    with pytest.raises(SystemExit) as excinfo:
        main(
            [f"{test_data_dir}/annot2_str.csv", f"{test_data_dir}/annot1.csv", "--plot"]
        )
    assert excinfo.value.code == 0


def test_annotation_stats_files():
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                f"{test_data_dir}/label.csv",
                f"{test_data_dir}/speaker.csv",
                "--files",
                f"{test_data_dir}/subset.txt",
            ]
        )
    assert excinfo.value.code == 0
