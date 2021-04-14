from pathlib import Path

import click
import netCDF4
import numpy as np

from emorec.dataset import write_features
from emorec.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("corpus", type=str)
def main(input: Path, corpus: str):
    """Converts auDeep dataset INPUT to our format. Also sets corpus
    attribute to CORPUS.
    """

    dataset = netCDF4.Dataset(input, "a")
    names = [Path(x).stem for x in dataset.variables["filename"]]
    dataset.variables["filename"][:] = np.array(names)
    features = np.array(dataset.variables["features"])
    dataset.close()

    feature_names = [f"audeep{i + 1}" for i in range(features.shape[1])]
    write_features(
        input,
        names=names,
        features=features,
        corpus=corpus,
        feature_names=feature_names,
    )
    print(f"Wrote netCDF4 file to {input}")


if __name__ == "__main__":
    main()
