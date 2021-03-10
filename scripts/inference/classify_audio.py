"""Perform inference on a given dataset using a pre-trained model.
Return the predicted class and the estimated confidence information.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from emorec.dataset import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help="Input data to predict on.")
    parser.add_argument('output', type=Path, help="Output.")
    parser.add_argument('--model', type=Path, required=True,
                        help="Pickled model.")
    args = parser.parse_args()

    dataset = Dataset(args.input)
    names = np.array(dataset.names)
    dataset.normalise()
    with open(args.model, 'rb') as fid:
        clf = pickle.load(fid)

    pred = clf.predict_proba(dataset.x)
    sort = np.argsort(pred[:, 0])[::-1]
    names = names[sort]
    prob = pred[sort, 0]

    df = pd.DataFrame({'Clip': names, 'Score': prob})
    df.to_csv(args.output, header=True, index=False)
    print(f"Wrote CSV to {args.output}")


if __name__ == "__main__":
    main()
