from pathlib import Path

import pytest
from click.testing import CliRunner

from ertk.cli.dataset.vis import main


def test_noargs():
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code == 2


def test_badargs():
    runner = CliRunner()
    result = runner.invoke(main, ["x", "y"])
    assert result.exit_code == 2


test_data_dir = Path(__file__).parent / "../../test_data"


def test_xy_noargs():
    runner = CliRunner()
    result = runner.invoke(main, ["xy"])
    assert result.exit_code == 2


def test_xy_data_conf(data_conf_file):
    runner = CliRunner()
    result = runner.invoke(main, ["xy", "--conf", str(data_conf_file)])
    assert result.exit_code == 0


def test_xy_data_corpus():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "xy",
            "--corpus",
            str(test_data_dir / "corpus_info.yaml"),
            "--features",
            str(test_data_dir / "features/features_2d.nc"),
        ],
    )
    assert result.exit_code == 0


def test_xy_data_corpus_no_features():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "xy",
            "--corpus",
            str(test_data_dir / "corpus_info.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_xy_data_features():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "xy",
            "--features",
            str(test_data_dir / "features/features_2d.nc"),
        ],
    )
    assert result.exit_code == 0


def test_xy_subsample_float():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "xy",
            "--features",
            str(test_data_dir / "features/features_2d.nc"),
            "--sample",
            "0.5",
        ],
    )
    assert result.exit_code == 0


def test_xy_subsample_int():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "xy",
            "--features",
            str(test_data_dir / "features/features_2d.nc"),
            "--sample",
            "6",
        ],
    )
    assert result.exit_code == 0


def test_xy_subsample_hue():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "xy",
            "--corpus",
            str(test_data_dir / "corpus_info.yaml"),
            "--features",
            str(test_data_dir / "features/features_2d.nc"),
            "--sample",
            "0.5",
            "--colour",
            "speaker",
        ],
    )
    assert result.exit_code == 0


@pytest.mark.parametrize("trans", ["pca", "tsne", "minmax", "std", "norm", "umap"])
def test_xy_transform(trans: str):
    if trans == "tsne":
        pytest.xfail("This test is expected to fail for 'tsne'")
    elif trans == "umap":
        pytest.importorskip("umap")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "xy",
            "--features",
            str(test_data_dir / "features/features_2d.nc"),
            "--transform",
            trans,
        ],
    )
    assert result.exit_code == 0


@pytest.mark.filterwarnings("ignore:non-interactive")
@pytest.mark.usefixtures("mpl")
def test_feats_int():
    runner = CliRunner()
    result = runner.invoke(
        main, ["feats", f"{test_data_dir}/features/features_3d.nc", "1"]
    )
    assert result.exit_code == 0


@pytest.mark.filterwarnings("ignore:non-interactive")
@pytest.mark.usefixtures("mpl")
def test_feats_slice():
    runner = CliRunner()
    result = runner.invoke(
        main, ["feats", f"{test_data_dir}/features/features_2d.nc", "1:4"]
    )
    assert result.exit_code == 0


@pytest.mark.filterwarnings("ignore:non-interactive")
@pytest.mark.usefixtures("mpl")
def test_feats_name():
    runner = CliRunner()
    result = runner.invoke(
        main, ["feats", f"{test_data_dir}/features/features_3d.nc", "1001_DFA_FEA_XX"]
    )
    assert result.exit_code == 0
