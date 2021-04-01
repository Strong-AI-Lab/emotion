import os
import subprocess
import tempfile
from pathlib import Path

import click
import netCDF4
import numpy as np
import pandas as pd

from emorec.dataset import write_netcdf_dataset
from emorec.utils import PathlibPath

OPENXBOW_JAR = "third_party/openxbow/openXBOW.jar"


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("output", type=Path)
@click.option(
    "--codebook",
    type=int,
    default=500,
    help="Size of codebook (number of outptut features).",
)
@click.option(
    "--closest",
    type=int,
    default=200,
    help="Number of closest codes to increment per vector. This acts as a "
    "smoothing parameter.",
)
def main(input: Path, output: Path, codebook: int, closest: int):
    """Process sequences of LLD vectors using the openXBOW software.
    INPUT should be a dataset containing the feature vectors, one per
    instance. The vector-quantized features are written to a new dataset
    in OUTPUT.
    """

    _, tmpin = tempfile.mkstemp(prefix="openxbow_", suffix=".csv")
    _, tmpout = tempfile.mkstemp(prefix="openxbow_", suffix=".csv")

    # We need to temporarily convert to CSV
    dataset = netCDF4.Dataset(input)
    corpus = dataset.corpus
    names = np.array(dataset.variables["name"])
    slices = np.array(dataset.variables["slices"])
    names = np.repeat(names, slices)
    features = np.array(dataset.variables["features"])
    n_features = features.shape[1]
    df = pd.concat([pd.Series(names), pd.DataFrame(features)], axis=1)
    df.to_csv(tmpin, header=False, index=False)
    dataset.close()

    attr_format = f"n1[{n_features}]"
    xbow_args = [
        "java",
        "-jar",
        f"{OPENXBOW_JAR}",
        "-i",
        tmpin,
        "-o",
        tmpout,
        "-attributes",
        attr_format,
        "-csvSep",
        ",",
        "-writeName",
        "-noLabels",
        "-size",
        str(codebook),
        "-a",
        str(closest),
        "-log",
        "-norm",
        "1",
    ]
    subprocess.call(xbow_args)
    os.remove(tmpin)

    data = pd.read_csv(tmpout, header=None, quotechar="'", dtype={0: str})
    os.remove(tmpout)

    write_netcdf_dataset(
        output,
        corpus=corpus,
        names=list(data.iloc[:, 0]),
        features=np.array(data.iloc[:, 1:]),
    )
    print(f"Wrote netCDF dataset to {output}")


if __name__ == "__main__":
    main()
