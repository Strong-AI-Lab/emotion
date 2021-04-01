from collections import Counter
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from emorec.dataset import write_netcdf_dataset
from emorec.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("corpus", type=str)
@click.argument("output", type=Path, required=False)
@click.option("--noheader", is_flag=True, help="Header information isn't present.")
@click.option("--nolabel", is_flag=True, help="No label column.")
def main(
    input: Path, corpus: str, output: Optional[Path], noheader: bool, nolabel: bool
):
    """Converts INPUT CSV file to a netCDF4 dataset at OUTPUT."""

    df = pd.read_csv(input, header=None if noheader else 0, converters={0: str})
    name_counts = Counter(df[df.columns[0]].map(lambda x: Path(x).stem))
    features = df.iloc[:, 1:-1]
    if nolabel:
        features = df.iloc[:, 1:]

    feature_names = []
    if not noheader:
        feature_names = df.columns[1:-1]
        if nolabel:
            feature_names = df.columns[1:]

    if output is None:
        output = input.with_suffix(".nc")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_netcdf_dataset(
        output,
        list(name_counts.keys()),
        features.to_numpy(),
        corpus=corpus,
        slices=list(name_counts.values()),
        feature_names=feature_names,
    )
    print(f"Wrote netCDF4 dataset to {output}")


if __name__ == "__main__":
    main()
