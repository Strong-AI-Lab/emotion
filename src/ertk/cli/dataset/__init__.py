import click

from .annotation_stats import main as annotation_stats_main
from .combine import main as combine_main
from .convert import main as convert_main
from .info import main as info_main
from .process import main as process_main
from .remove_instances import main as remove_instances_main
from .view_features import main as view_features_main


@click.group(no_args_is_help=True)
def cli():
    """ERTK dataset-related scripts."""
    click.echo("ERTK dataset CLI")


cli.add_command(annotation_stats_main, "annotation")
cli.add_command(combine_main, "combine")
cli.add_command(convert_main, "convert")
cli.add_command(info_main, "info")
cli.add_command(process_main, "process")
cli.add_command(remove_instances_main, "remove_instances")
cli.add_command(view_features_main, "view")
