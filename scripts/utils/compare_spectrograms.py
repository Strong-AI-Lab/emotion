#!/usr/bin/python3

import argparse

import numpy as np
import netCDF4
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file1')
    parser.add_argument('file2')
    args = parser.parse_args()

    dataset = netCDF4.Dataset(args.file1)
    spec1 = np.array(dataset.variables['features'])
    filenames1 = list(dataset.variables['filename'])
    dataset.close()

    dataset = netCDF4.Dataset(args.file2)
    spec2 = np.array(dataset.variables['features'])
    filenames2 = list(dataset.variables['filename'])
    filenames2 = [x[:-4] for x in filenames2]
    sort_idx = np.argsort(filenames2)
    spec2 = spec2[sort_idx]
    filenames2 = sorted(filenames2)
    dataset.close()

    assert set(filenames1) == set(filenames2)

    diff = np.mean((spec1 - spec2)**2, axis=(1, 2))
    print("Differences: {}, {}".format(diff.mean(), diff.max()))
    vis = np.argmax(diff)

    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.suptitle('Spectrogram visualisation')
    ax[0].imshow(spec1[vis])
    ax[0].set_title(filenames1[vis])
    ax[1].imshow(spec2[vis])
    ax[1].set_title(filenames2[vis])
    plt.show()


if __name__ == "__main__":
    main()
