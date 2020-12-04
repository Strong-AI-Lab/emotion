"""Combines two or more datasets of spectrograms into a larger dataset.
"""

import argparse
from pathlib import Path

import netCDF4
import numpy as np
from emotion_recognition.dataset import write_netcdf_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='+', type=Path,
                        help="Input files in netCDF4 format.")
    parser.add_argument('output_file', type=Path, help="Output dataset.")
    args = parser.parse_args()

    features = []
    names = []
    slices = []
    labels = []
    num_features = None
    for filename in args.input_files:
        print("Opened netCDF4 dataset {}".format(str(filename)))
        data = netCDF4.Dataset(str(filename))
        if not num_features:
            num_features = data.variables['features'].shape[-1]
        elif data.variables['features'].shape[-1] != num_features:
            raise ValueError("Feature size of all datasets must match.")

        features.append(np.array(data.variables['features']))
        names.append(np.array(data.variables['filename']))
        slices.append(np.array(data.variables['slices']))
        # TODO: Make it also work for numeric annotations
        labels.append(np.array(data.variables['label_nominal']))
        data.close()
    features = np.concatenate(features)
    names = np.concatenate(names)
    slices = np.concatenate(slices)
    labels = np.concatenate(labels)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    write_netcdf_dataset(
        args.output_file, corpus='combined', names=names, features=features,
        slices=slices, annotations=labels, annotation_type='classification'
    )
    print("Wrote netCDF4 dataset to {}".format(args.output_file))


if __name__ == "__main__":
    main()
