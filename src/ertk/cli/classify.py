import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd

from ertk.dataset import read_features


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=Path)
@click.option(
    "--model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Pickled model.",
)
@click.option(
    "--norm",
    type=click.Choice(["speaker", "corpus", "all"]),
    default="speaker",
    show_default=True,
    help="Normalisation scheme.",
)
def main(input: Path, output: Path, model: Path, norm: str):
    """Perform inference on given INPUT features using a pre-trained
    model. Gives the predicted class and confidence and writes CSV to
    OUTPUT.
    """

    dataset = read_features(input)
    names = np.array(dataset.names)
    with open(model, "rb") as fid:
        print(f"Loading model from {model}")
        clf = pickle.load(fid)

    print(f"Running inference on {len(dataset)} clips")
    if hasattr(clf, "predict_proba"):
        pred = clf.predict_proba(dataset.features)
        labels = np.argmax(pred, axis=-1)
        scores = pred[np.arange(pred.shape[0]), labels]
    else:
        labels = clf.predict(dataset.features)
        scores = np.zeros(len(names))

    output.parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame({"Clip": names, "Label": labels, "Score": scores})
    df.to_csv(output, header=True, index=False)
    print(f"Wrote CSV to {output}")


if __name__ == "__main__":
    main()
