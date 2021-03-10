from pathlib import Path
from typing import Tuple

import click
import netCDF4
import numpy as np
from emorec.dataset import write_netcdf_dataset
from emorec.utils import PathlibPath


@click.command()
@click.argument('input', type=PathlibPath(exists=True, dir_okay=False),
                nargs=-1)
@click.argument('output', type=Path)
def main(input: Tuple[Path], output: Path):
    """Combines multiple INPUT datasets of spectrograms into a larger
    dataset saved to OUTPUT.
    """

    if len(input) == 0:
        raise ValueError("No input files specified.")

    total_length = 0
    total_instances = 0
    num_features = 0
    feature_names = []
    for filename in input:
        data = netCDF4.Dataset(str(filename))
        if num_features == 0:
            num_features = len(data.dimensions['features'])
        elif len(data.dimensions['features']) != num_features:
            raise ValueError("Feature size of all datasets must match.")
        total_length += len(data.dimensions['concat'])
        total_instances += len(data.dimensions['instance'])
        feature_names = list(data.variables['feature_names'])
        data.close()

    # Preallocate arrays to save storing as lists in memory
    features = np.empty((total_length, num_features), dtype=np.float32)
    names = np.empty(total_instances, dtype=str)
    slices = np.empty(total_instances, dtype=int)
    l_idx = 0
    i_idx = 0
    for filename in input:
        print(f"Opened netCDF4 dataset {filename}")
        data = netCDF4.Dataset(str(filename))
        length = len(data.dimensions['concat'])
        instances = len(data.dimensions['instance'])
        features[l_idx:l_idx + length, :] = np.array(
            data.variables['features'])
        names[i_idx:i_idx + instances] = np.array(data.variables['name'])
        slices[i_idx:i_idx + instances] = np.array(data.variables['slices'])
        l_idx += length
        i_idx += instances
        data.close()
    assert l_idx == total_length and i_idx == total_instances

    output.parent.mkdir(parents=True, exist_ok=True)
    write_netcdf_dataset(output, corpus='combined', names=names, slices=slices,
                         features=features, feature_names=feature_names)
    print(f"Wrote netCDF4 dataset to {output}")


if __name__ == "__main__":
    main()
