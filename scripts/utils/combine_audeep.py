"""Combines two or more datasets of spectrograms into a larger dataset.
"""

import argparse
from pathlib import Path

import netCDF4
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='+', type=Path,
                        help="Input files in auDeep format.")
    parser.add_argument('output_file', type=Path, help="Output dataset.")
    args = parser.parse_args()

    all_features = []
    all_names = []
    shape = None
    for filename in args.input_files:
        print(f"Opened netCDF4 dataset {filename}")
        data = netCDF4.Dataset(str(filename))
        if not shape:
            shape = data.variables['features'].shape[1:]
        elif data.variables['features'].shape[1:] != shape:
            raise ValueError("Shapes of spectrograms in all datasets must "
                             "match.")

        all_features.append(np.array(data.variables['features']))
        all_names.append(np.array(data.variables['filename']))
        data.close()
    all_features = np.concatenate(all_features)
    all_names = np.concatenate(all_names)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = netCDF4.Dataset(str(args.output_file), 'w')
    dataset.createDimension('instance', len(all_names))
    dataset.createDimension('fold', 0)
    dataset.createDimension('time', shape[0])
    dataset.createDimension('freq', shape[1])

    filename = dataset.createVariable('filename', str, ('instance',))
    filename[:] = all_names
    features = dataset.createVariable('features', np.float32,
                                      ('instance', 'time', 'freq'))
    features[:, :, :] = all_features.astype(np.float32)
    dataset.setncattr_string('feature_dims', '["time", "freq"]')
    dataset.setncattr_string('corpus', '')

    # Unused vars
    chunk_nr = dataset.createVariable('chunk_nr', np.int64, ('instance',))
    chunk_nr[:] = np.zeros(len(all_names), dtype=np.int64)
    partition = dataset.createVariable('partition', np.float64, ('instance',))
    partition[:] = np.zeros(len(all_names), dtype=np.float64)
    dataset.createVariable('cv_folds', np.int64, ('instance', 'fold'))
    label_nominal = dataset.createVariable('label_nominal', str, ('instance',))
    label_nominal[:] = np.zeros(len(all_names), dtype=str)
    label_numeric = dataset.createVariable('label_numeric', np.int64,
                                           ('instance',))
    label_numeric[:] = np.zeros(len(all_names), dtype=np.int64)

    dataset.close()

    print(f"Wrote netCDF4 dataset to {args.output_file}")


if __name__ == "__main__":
    main()
