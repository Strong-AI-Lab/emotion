"""
``ertk-dataset`` scripts
========================

This module contains the ``ertk-dataset`` scripts.

.. autosummary::
    :toctree:

    cli


Commands
--------
.. autosummary::
    :toctree:

    annotation_stats_main
    combine_main
    convert_main
    filter_clips_main
    info_main
    process_main
    remove_instances_main
    vis_main
"""

import click

from .annotation_stats import main as annotation_stats_main
from .combine import main as combine_main
from .convert import main as convert_main
from .filter_clips import main as filter_clips_main
from .info import main as info_main
from .process import main as process_main
from .remove_instances import main as remove_instances_main
from .vis import main as vis_main


@click.group(no_args_is_help=True)
def cli():
    """ERTK dataset-related scripts."""
    click.echo("ERTK dataset CLI")


cli.add_command(annotation_stats_main, "annotation")
cli.add_command(combine_main, "combine")
cli.add_command(convert_main, "convert")
cli.add_command(filter_clips_main, "filter_clips")
cli.add_command(info_main, "info")
cli.add_command(process_main, "process")
cli.add_command(remove_instances_main, "remove_instances")
cli.add_command(vis_main, "vis")
