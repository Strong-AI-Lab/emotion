import pytest

from ertk.cli.cli import cli as cli_cli
from ertk.cli.dataset import cli as cli_dataset
from ertk.cli.util import cli as cli_util


@pytest.mark.parametrize("cli", [cli_cli, cli_dataset, cli_util])
def test_noargs(cli):
    with pytest.raises(SystemExit) as excinfo:
        cli([])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("cli", [cli_cli, cli_dataset, cli_util])
def test_bad_command(cli):
    with pytest.raises(SystemExit) as excinfo:
        cli(["xyzabc"])
    assert excinfo.value.code == 2
