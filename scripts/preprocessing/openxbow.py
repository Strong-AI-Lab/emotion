import os
import subprocess
import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd

from ertk.dataset import read_features, write_features
from ertk.utils import PathlibPath


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
@click.option(
    "--jar",
    type=PathlibPath(exists=True),
    default=Path("third_party/openxbow/openXBOW.jar"),
    help="Path to openXBOW.jar",
    show_default=True,
)
@click.option("--log/--nolog", default=True, help="Use log(1 + x) scaling.")
def main(input: Path, output: Path, codebook: int, closest: int, jar: Path, log: bool):
    """Process sequences of LLD vectors using the openXBOW software.
    INPUT should be a dataset containing the feature vectors, one per
    instance. The vector-quantized features are written to a new dataset
    in OUTPUT.
    """

    _, tmpin = tempfile.mkstemp(prefix="openxbow_", suffix=".csv")
    _, tmpout = tempfile.mkstemp(prefix="openxbow_", suffix=".csv")

    dataset = read_features(input)
    dataset.write_csv(tmpin, header=False)

    xbow_args = [
        "java",
        "-jar",
        str(jar),
        "-i",
        tmpin,
        "-o",
        tmpout,
        "-attributes",
        f"n1[{len(dataset.feature_names)}]",
        "-csvSep",
        ",",
        "-writeName",
        "-noLabels",
        "-size",
        str(codebook),
        "-a",
        str(closest),
        "-norm",
        "1",
    ]
    if log:
        xbow_args.append("-log")
    subprocess.call(xbow_args)
    os.remove(tmpin)
    data = pd.read_csv(tmpout, header=None, quotechar="'", dtype={0: str})
    os.remove(tmpout)

    write_features(
        output,
        corpus=dataset.corpus,
        names=list(data.iloc[:, 0]),
        features=np.array(data.iloc[:, 1:]),
    )
    print(f"Wrote dataset to {output}")


if __name__ == "__main__":
    main()
