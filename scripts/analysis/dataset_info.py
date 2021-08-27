from pathlib import Path
from typing import Tuple

import click

from ertk.dataset import Dataset
from ertk.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False), nargs=-1)
@click.option("--subset", default="default")
def main(input: Tuple[Path], subset: str):
    """Print info about datasets in INPUT."""
    for file in input:
        dataset = Dataset(file, subset=subset)
        print(dataset)


if __name__ == "__main__":
    main()
