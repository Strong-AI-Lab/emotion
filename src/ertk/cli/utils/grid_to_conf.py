from pathlib import Path

import click
import yaml
from sklearn.model_selection import ParameterGrid

from ertk.config import get_arg_mapping


@click.command()
@click.argument(
    "param_grid", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output", type=Path)
@click.option("--format", help="Format string.")
def main(param_grid: Path, output: Path, format: str):
    """Creates a new parameters YAML file in the OUTPUT directory for
    each combination of parameters in the PARAM_GRID file. The names of
    the files will be formatted according to the --format parameter if
    given, or else assigned a number starting from 1.
    """
    grid = get_arg_mapping(param_grid)
    output.mkdir(exist_ok=True, parents=True)
    for i, params in enumerate(ParameterGrid(grid)):
        if format:
            filename = format.format(**params)
        else:
            filename = f"params_{i:02d}"
        if not filename.endswith(".yaml"):
            filename += ".yaml"
        with open(output / filename, "w") as fid:
            yaml.dump(params, fid)
            print(f"Wrote {output / filename}.")


if __name__ == "__main__":
    main()
