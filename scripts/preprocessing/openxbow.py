import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd

from ertk.dataset import read_features, write_features
from ertk.utils import PathlibPath

OPENXBOW = Path(__file__).parent / "../../third_party/openxbow/openXBOW.jar"


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("output", type=Path)
@click.option(
    "--jar",
    type=PathlibPath(exists=True),
    default=OPENXBOW.resolve(),
    help="Path to openXBOW.jar",
    show_default=True,
)
@click.argument("xbowargs", nargs=-1)
def main(input: Path, output: Path, jar: Path, xbowargs: Tuple[str]):
    """Process sequences of LLD vectors using the openXBOW software.
    INPUT should be the frame-level feature vectors. The BoAW from
    vector-quantised features are written to OUTPUT.
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
        *xbowargs,
    ]
    print(f"Using SMILExtract at {jar} with extra options:\n\t{' '.join(xbowargs)}")
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
