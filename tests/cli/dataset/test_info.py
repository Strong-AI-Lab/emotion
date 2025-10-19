from pathlib import Path

from click.testing import CliRunner

from ertk.cli.dataset.info import main


def test_noargs():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 2
    result.return_value
    assert "No corpus info files" in result.output


def test_badargs():
    runner = CliRunner()
    result = runner.invoke(main, ["x", "y"])
    assert result.exit_code == 2
    assert "Error" in result.output


test_data_dir = Path(__file__).parent / "../../test_data"


def test_info():
    runner = CliRunner()
    result = runner.invoke(main, [f"{test_data_dir}/corpus_info.yaml"])
    assert result.exit_code == 0


def test_verbose():
    runner = CliRunner()
    result = runner.invoke(
        main, [f"{test_data_dir}/corpus_info.yaml", "--verbose", "1"]
    )
    assert result.exit_code == 0


def test_subset():
    runner = CliRunner()
    result = runner.invoke(
        main, [f"{test_data_dir}/corpus_info.yaml", "--subset", "subset1"]
    )
    assert result.exit_code == 0


def test_subset_map():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [f"{test_data_dir}/corpus_info.yaml", "--subset", "test_corpus:subset1"],
    )
    assert result.exit_code == 0


def test_map_groups():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            f"{test_data_dir}/corpus_info.yaml",
            "--map_groups",
            f"{test_data_dir}/speaker_map_test.yaml",
        ],
    )
    assert result.exit_code == 0


def test_sel_groups():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [f"{test_data_dir}/corpus_info.yaml", "--sel_groups", "speaker:1001"],
    )
    assert result.exit_code == 0


def test_remove_groups():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            f"{test_data_dir}/corpus_info.yaml",
            "--remove_groups",
            "speaker:1002",
        ],
    )
    assert result.exit_code == 0


def test_clip_groups():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [f"{test_data_dir}/corpus_info.yaml", "--clip_seq", "100"],
    )
    assert result.exit_code == 0


def test_pad_groups():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [f"{test_data_dir}/corpus_info.yaml", "--clip_seq", "100"],
    )
    assert result.exit_code == 0


def test_data_config(data_conf_file):
    runner = CliRunner()
    result = runner.invoke(
        main, ["--data_config", data_conf_file], catch_exceptions=False
    )
    print(result.output)
    assert not result.exception
    assert result.exit_code == 0


def test_output_list_config(tmp_path, data_conf_file):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--data_config",
            data_conf_file,
            "--output_list",
            tmp_path / "test_list.txt",
        ],
    )
    assert result.exit_code == 0
    assert Path(tmp_path / "test_list.txt").exists()


def test_output_list(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            f"{test_data_dir}/corpus_info.yaml",
            "--output_list",
            tmp_path / "test_list.txt",
        ],
    )
    assert result.exit_code == 0
    assert Path(tmp_path / "test_list.txt").exists()
