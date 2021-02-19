"""Process sequences of LLD vectors using the openXBOW software."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from emorec.dataset import write_netcdf_dataset

OPENXBOW_JAR = 'third_party/openxbow/openXBOW.jar'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True,
                        help="Input netCDF4 file.")
    parser.add_argument('--output', type=Path, required=True,
                        help="Output CSV file.")
    parser.add_argument('--labels', required=True, type=Path,
                        help="Path to labels CSV.")
    parser.add_argument('--codebook', type=int, default=500,
                        help="Size of codebook (number of outptut features).")
    parser.add_argument(
        '--closest', type=int, default=200,
        help="Number of closest codes to increment per vector. This acts as a "
        "smoothing parameter."
    )
    args = parser.parse_args()

    _, tmpin = tempfile.mkstemp(prefix='openxbow_', suffix='.csv')
    _, tmpout = tempfile.mkstemp(prefix='openxbow_', suffix='.csv')

    # We need to temporarily convert to CSV
    dataset = netCDF4.Dataset(args.input)
    corpus = dataset.corpus
    names = np.array(dataset.variables['filename'])
    slices = np.array(dataset.variables['slices'])
    names = np.repeat(names, slices)
    features = np.array(dataset.variables['features'])
    n_features = features.shape[1]
    df = pd.concat([pd.Series(names), pd.DataFrame(features)], axis=1)
    df.to_csv(tmpin, header=False, index=False)
    dataset.close()

    attr_format = 'n1[{}]'.format(n_features)
    xbow_args = [
        'java', '-jar', '{}'.format(OPENXBOW_JAR),
        '-i', tmpin,
        '-o', tmpout,
        '-attributes', attr_format,
        '-csvSep', ',',
        '-writeName',
        '-noLabels',
        '-size', str(args.codebook),
        '-a', str(args.closest),
        '-log',
        '-norm', '1'
    ]
    subprocess.call(xbow_args)
    os.remove(tmpin)

    data = pd.read_csv(tmpout, header=None, quotechar="'")
    os.remove(tmpout)

    write_netcdf_dataset(
        args.output, corpus=corpus, names=list(data.iloc[:, 0]),
        slices=np.ones(len(data)), features=np.array(data.iloc[:, 1:]),
        annotation_path=args.labels, annotation_type='classification'
    )
    print("Wrote netCDF dataset to {}.".format(args.output))


if __name__ == "__main__":
    main()
