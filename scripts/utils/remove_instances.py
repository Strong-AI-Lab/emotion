from pathlib import Path

import click
import numpy as np

from ertk.dataset import read_features, write_features
from ertk.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("output", type=Path, required=False)
@click.option("--names", type=PathlibPath(exists=True, dir_okay=False))
def main(input: Path, output: Path, names: Path):
    """Remove instances from INPUT features that aren't in the NAMES
    file, and write to OUTPUT. If OUTPUT is not given, overwrites INPUT
    in-place.
    """
    with open(names) as fid:
        keep_names = {Path(x.strip()).stem for x in fid}
    data = read_features(input)
    idx = np.array([i for i, n in enumerate(data.names) if n in keep_names])

    if not output:
        output = input
    write_features(
        output,
        corpus=data.corpus,
        names=data.names[idx],
        features=data.features[idx],
        feature_names=data.feature_names,
    )
    print(f"Wrote features to {input}")


if __name__ == "__main__":
    main()
