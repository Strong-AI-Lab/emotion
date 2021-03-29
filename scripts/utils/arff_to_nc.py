from collections import Counter
from pathlib import Path
from typing import Optional

import arff
import click
import numpy as np
from emorec.dataset import write_netcdf_dataset
from emorec.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("output", type=Path, required=False)
def main(input: Path, output: Optional[Path]):
    """Converts INPUT ARFF file to a netCDF4 dataset at OUTPUT."""

    print("Reading")
    with open(input, "r") as fid:
        data = arff.load(fid)

    counts = Counter([x[0] for x in data["data"]])
    features = np.array([x[1:-1] for x in data["data"]])
    feature_names = [x[0] for x in data["attributes"][1:-1]]
    if output is None:
        output = input.with_suffix(".nc")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_netcdf_dataset(
        output,
        list(counts.keys()),
        features,
        slices=list(counts.values()),
        feature_names=feature_names,
    )
    print(f"Wrote netCDF4 dataset to {output}")


if __name__ == "__main__":
    main()
