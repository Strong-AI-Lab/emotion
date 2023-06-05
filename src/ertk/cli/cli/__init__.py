"""
``ertk-cli`` scripts
====================

This module contains the ``ertk-cli`` scripts.

.. autosummary::
    :toctree:

    cli


Commands
--------
.. autosummary::
    :toctree:

    classify_main
    exp_main
    exp2_main
"""

import click

from .classify import main as classify_main
from .exp import main as exp_main
from .exp2 import main as exp2_main


@click.group(no_args_is_help=True)
def cli():
    """ERTK command-line scripts."""
    click.echo("ERTK CLI")


cli.add_command(classify_main, "classify")
cli.add_command(exp_main, "exp")
cli.add_command(exp2_main, "exp2")
