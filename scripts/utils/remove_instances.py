from pathlib import Path

import click
import netCDF4
import numpy as np
from emorec.dataset import write_netcdf_dataset
from emorec.utils import PathlibPath


@click.command()
@click.argument('input', type=PathlibPath(exists=True, dir_okay=False))
@click.argument('names', type=PathlibPath(exists=True, dir_okay=False))
def main(input: Path, names: Path):
    """Removes instances from INPUT dataset that aren't in the NAMES
    file. INPUT is modified in-place.

    This script is mainly useful if a feature generation script
    generates features for all audio files instead of just the ones in
    a file list.
    """

    data = netCDF4.Dataset(str(input))
    slices = np.array(data.variables['slices'])
    _names = np.array(data.variables['name'])
    features = np.array(data.variables['features'])
    feature_names = list(data.variables['feature_names'])
    corpus = data.corpus
    data.close()

    with open(names) as fid:
        keep_names = {Path(x.strip()).stem for x in fid}
    idx = np.array([i for i, n in enumerate(_names) if n in keep_names])
    new_slices = slices[idx]
    new_names = _names[idx]
    cum_slices = np.r_[0, np.cumsum(slices)]
    feat_idx = np.concatenate([np.arange(cum_slices[i], cum_slices[i + 1])
                               for i in idx])
    new_features = features[feat_idx]

    write_netcdf_dataset(
        input, corpus=corpus, names=new_names, slices=new_slices,
        features=new_features, feature_names=feature_names
    )
    print(f"Wrote netCDF4 dataset to {input}")


if __name__ == "__main__":
    main()
