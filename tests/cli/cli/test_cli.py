import pytest

from ertk.cli.cli import cli


def test_cli():
    with pytest.raises(SystemExit) as excinfo:
        cli([])
    assert excinfo.value.code == 2
