from pathlib import Path

import pandas as pd
import pytest

from ertk.dataset.annotation import read_annotations, write_annotations

from .constants import test_data_dir


def test_read_annotations_str_1() -> None:
    annotations = read_annotations(test_data_dir / "annot2_str.csv", dtype=str)
    assert type(annotations["extra1"]) is str
    assert annotations["extra1"] == "1"
    assert annotations["extra2"] == "0"
    assert annotations["1002_DFA_DIS_XX"] == "1"
    assert annotations["1002_DFA_HAP_XX"] == "3"


def test_read_annotations_str_2() -> None:
    annotations = read_annotations(test_data_dir / "label.csv", dtype=str)
    assert type(annotations["extra1"]) is str
    assert annotations["extra1"] == "none"
    assert annotations["extra2"] == "sadness"
    assert annotations["1002_DFA_DIS_XX"] == "disgust"
    assert annotations["1002_DFA_HAP_XX"] == "happiness"


def test_read_annotations_float() -> None:
    annotations = read_annotations(test_data_dir / "annot1.csv", dtype=float)
    assert type(annotations["extra1"]) is float
    assert annotations["extra1"] == 0
    assert annotations["extra2"] == 1
    assert annotations["1001_DFA_HAP_XX"] == pytest.approx(3.8472069)
    assert annotations["1001_DFA_NEU_XX"] == pytest.approx(6.9999)


def test_write_annotations_str(tmp_path: Path) -> None:
    mapping = {"clip1": "x", "clip2": "y", "clip3": "z"}
    out_path = tmp_path / "annotaton.csv"
    write_annotations(mapping, "test_annot", out_path)
    df = pd.read_csv(out_path, header=0, index_col=0, dtype=str)
    assert list(df.index) == ["clip1", "clip2", "clip3"]
    assert df.columns[0] == "test_annot"
    assert df.loc["clip1", "test_annot"] == "x"


def test_write_annotations_float(tmp_path: Path) -> None:
    mapping = {"clip1": 1.5, "clip2": 2, "clip3": 0.9}
    out_path = tmp_path / "annotaton_float.csv"
    write_annotations(mapping, "test_annot2", out_path)
    df = pd.read_csv(out_path, header=0, index_col=0, dtype={0: str, 1: float})
    assert list(df.index) == ["clip1", "clip2", "clip3"]
    assert df.columns[0] == "test_annot2"
    assert df.loc["clip2", "test_annot2"] == 2
    assert df.loc["clip3", "test_annot2"] == pytest.approx(0.9)
