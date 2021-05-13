import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd

from emorec.dataset import Dataset
from emorec.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True))
@click.argument("output", type=Path)
@click.option(
    "--model",
    type=PathlibPath(exists=True, dir_okay=False),
    required=True,
    help="Pickled model.",
)
@click.option(
    "--norm",
    type=click.Choice(["speaker", "corpus", "all"]),
    default="speaker",
    help="Normalisation scheme.",
)
def main(input: Path, output: Path, model: Path, norm: str):
    """Perform inference on a given INPUT features using a pre-trained
    model. Gives the predicted class and confidence and writes CSV to
    OUTPUT.
    """

    print(f"Reading features from {input}")
    dataset = Dataset(input)
    names = np.array(dataset.names)
    dataset.normalise(scheme=norm)
    with open(model, "rb") as fid:
        print(f"Loading model from {model}")
        clf = pickle.load(fid)

    print(f"Running inference on {len(dataset)} clips")
    pred = clf.predict_proba(dataset.x)
    sort = np.argsort(pred[:, 0])[::-1]
    names = names[sort]
    prob = pred[sort, 0]

    output.parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame({"Clip": names, "Score": prob})
    df.to_csv(output, header=True, index=False)
    print(f"Wrote CSV to {output}")


if __name__ == "__main__":
    main()
