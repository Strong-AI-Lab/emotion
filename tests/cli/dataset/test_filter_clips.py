from pathlib import Path

from click.testing import CliRunner

from ertk.cli.dataset.filter_clips import main


def test_noargs():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 2
    result.return_value
    assert "Missing argument" in result.output


def test_badargs():
    runner = CliRunner()
    result = runner.invoke(main, ["x", "y"])
    assert result.exit_code == 2
    assert "Error" in result.output


test_data_dir = Path(__file__).parent / "../../test_data"


def test_input_dir(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main, [str(test_data_dir / "data"), str(tmp_path / "output.txt")]
    )
    assert result.exit_code == 0


def test_input_file(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main, [str(test_data_dir / "all_clips.txt"), str(tmp_path / "output.txt")]
    )
    assert result.exit_code == 0


def test_length_limits(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            str(test_data_dir / "all_clips.txt"),
            str(tmp_path / "output.txt"),
            "--minlength",
            "1",
            "--maxlength",
            "2.5",
        ],
    )
    assert result.exit_code == 0
    assert "Found 8 valid clips" in result.output
    with open(tmp_path / "output.txt") as fid:
        assert len(list(fid)) == 8  # 8 valid clips


def test_invalid_limits(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            str(test_data_dir / "all_clips.txt"),
            str(tmp_path / "output.txt"),
            "--minlength",
            "10",
            "--maxlength",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "Found 0 valid clips" in result.output
