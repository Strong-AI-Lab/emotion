from pathlib import Path

import pytest

from ertk.cli.cli.exp2 import main

_exp_yaml = Path(__file__).parent / "exp.yaml"


def test_noargs():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_badargs():
    with pytest.raises(SystemExit) as excinfo:
        main(["x", "y"])
    assert excinfo.value.code == 2


def test_exp2():
    with pytest.raises(SystemExit) as excinfo:
        main([str(_exp_yaml)])
    assert excinfo.value.code == 0
