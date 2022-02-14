import click

from .class_cv import main as class_cv_main
from .class_tvt import main as tvt_main
from .classify import main as classify_main


@click.group(invoke_without_command=True)
def cli():
    """ERTK command-line scripts."""
    click.echo("ERTK CLI")


cli.add_command(classify_main, "classify")
cli.add_command(class_cv_main, "class_cv")
cli.add_command(tvt_main, "class_tvt")
