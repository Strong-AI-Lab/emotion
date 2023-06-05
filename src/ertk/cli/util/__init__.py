"""
``ertk-util`` scripts
=====================

This module contains the ``ertk-util`` scripts.

.. autosummary::
    :toctree:

    cli


Commands
--------
.. autosummary::
    :toctree:

    convert_audeep_main
    create_cv_dirs_main
    grid_to_conf_main
    names_to_filenames_main
    parallel_jobs_main
    split_chat_main
    split_elan_main
"""

import click

from .convert_audeep import main as convert_audeep_main
from .create_cv_dirs import main as create_cv_dirs_main
from .grid_to_conf import main as grid_to_conf_main
from .names_to_filenames import main as names_to_filenames_main
from .parallel_jobs import main as parallel_jobs_main
from .split_chat import main as split_chat_main
from .split_elan import main as split_elan_main


@click.group(no_args_is_help=True)
def cli():
    """ERTK utility scripts."""
    click.echo("ERTK utility CLI")


cli.add_command(convert_audeep_main, "convert_audeep")
cli.add_command(create_cv_dirs_main, "create_cv_dirs")
cli.add_command(grid_to_conf_main, "grid_to_conf")
cli.add_command(parallel_jobs_main, "parallel_jobs")
cli.add_command(names_to_filenames_main, "names_to_filenames")
cli.add_command(split_chat_main, "split_chat")
cli.add_command(split_elan_main, "split_elan")
