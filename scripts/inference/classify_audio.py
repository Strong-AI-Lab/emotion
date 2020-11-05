#!/usr/bin/python3

"""Perform inference on a given dataset using a pre-trained model.
Return the predicted class and the estimated confidence information.
"""

import argparse
import pickle
from pathlib import Path

import netCDF4
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True,
                        help="Input data to predict on.")
    parser.add_argument('--model', type=Path, required=True,
                        help="Pickled model.")
    parser.add_argument('--output', type=Path, required=True,
                        help="Output.")
    args = parser.parse_args()

    dataset = netCDF4.Dataset(args.input)
    data = np.array(dataset.variables['features'])
    names = np.array(dataset.variables['filename'])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    with open(args.model, 'rb') as fid:
        clf = pickle.load(fid)

    pred = clf.predict_proba(data)
    top_emo = np.argsort(pred[:, 0])[::-1]
    names = names[top_emo]
    prob = pred[top_emo, 0]

    with open(args.output, 'w') as fid:
        for name, p in zip(names, prob):
            label = 'emotional' if p > 0.5 else 'neutral'
            print('{},{},{}'.format(name, label, p), file=fid)
        print("Wrote CSV to {}".format(args.output))


if __name__ == "__main__":
    main()
