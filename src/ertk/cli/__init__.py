import click

from ertk import __version__

from .process import process_main


@click.group(invoke_without_command=True)
def main():
    """ERTK CLI"""
    click.echo(f"ERTK CLI version {__version__}")


main.add_command(process_main, "process")
