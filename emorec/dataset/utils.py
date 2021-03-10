import subprocess
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Type, Union

import netCDF4
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..utils import PathOrStr


def parse_annotations(filename: PathOrStr,
                      dtype: Optional[Type] = None) -> Dict[str, str]:
    """Returns a dict of the form {name: annotation}."""
    # Need index_col to be False or None due to
    # https://github.com/pandas-dev/pandas/issues/9435
    df = pd.read_csv(filename, index_col=False, header=0,
                     converters={0: str, 1: dtype})
    type_ = df.columns[1]
    annotations = df.set_index('name')[type_].to_dict()
    return annotations


def get_audio_paths(file: PathOrStr) -> Sequence[Path]:
    """Given a path to a file containing a list of audio files, returns
    a sequence of absolute paths to the audio files.

    Args:
    -----
    file: pathlike or str
        Path to a file containing a list of paths to audio clips.

    Returns:
    --------
        Sequence of paths to audio files.
    """
    file = Path(file)
    paths = []
    with open(file) as fid:
        for line in fid:
            p = Path(line.strip())
            paths.append(p if p.is_absolute() else (file.parent / p).resolve())
    return paths


def write_netcdf_dataset(path: PathOrStr,
                         names: Union[Sequence[str], np.ndarray],
                         features: np.ndarray,
                         slices: Union[Sequence[int], np.ndarray] = [],
                         corpus: str = '',
                         feature_names: Sequence[str] = []):
    """Writes a netCDF4 dataset to the given path. The dataset should
    contain features and annotations. Note that the features matrix has
    to be 2-D, and can either be a vector per instance, or a sequence of
    vectors per instance. Also note that this cannot represent the
    spectrograms in the format required by auDeep, since that is a 3-D
    matrix of one spectrogram per instance.

    Args:
    -----
    path: pathlike or str
        The path to write the dataset.
    corpus: str
        The corpus name
    names: sequence of str
        Instance names
    features: ndarray
        Features matrix of shape (length, n_features).
    slices: list of int, optional
        The size of each slice along axis 0 of features. If there is one
        vector per instance, then this will be all 1's, otherwise will
        have the length of the sequence corresponding to each instance.
    """
    dataset = netCDF4.Dataset(path, 'w')
    dataset.createDimension('instance', len(names))
    dataset.createDimension('concat', features.shape[0])
    dataset.createDimension('features', features.shape[1])

    if len(slices) == 0:
        slices = np.ones(len(names))
    if sum(slices) != len(features):
        raise ValueError(f"Total slices is {sum(slices)} but length of "
                         f"features is {len(features)}.")
    _slices = dataset.createVariable('slices', int, ('instance',))
    _slices[:] = np.array(slices)

    filename = dataset.createVariable('name', str, ('instance',))
    filename[:] = np.array(names)

    _features = dataset.createVariable('features', np.float32,
                                       ('concat', 'features'))
    _features[:, :] = features

    if len(feature_names) < features.shape[1]:
        feature_names = [f'feature{i}' for i in range(features.shape[1])]
    _feature_names = dataset.createVariable('feature_names', str,
                                            ('features',))
    _feature_names[:] = np.array(feature_names)

    dataset.setncattr_string('corpus', corpus)
    dataset.close()


def resample_audio(paths: Iterable[Path], dir: PathOrStr):
    """Resample given audio clips to 16 kHz 16-bit WAV, and place in
    direcotory given by `dir`.
    """
    paths = list(paths)
    if len(paths) == 0:
        raise FileNotFoundError("No audio files found.")

    dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)
    print(f"Resampling {len(paths)} audio files to {dir}")

    print("Using FFmpeg options: -nostdin -ar 16000 -sample_fmt s16")
    Parallel(n_jobs=-1, verbose=1)(
        delayed(subprocess.run)(
            ['ffmpeg', '-nostdin', '-i', str(path), '-ar', '16000',
             '-sample_fmt', 's16', '-y', str(dir / (path.stem + '.wav'))],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        ) for path in paths
    )


def write_filelist(paths: Iterable[Path], out: PathOrStr = 'files.txt'):
    """Write sorted file list."""
    paths = sorted(paths, key=lambda p: p.stem)
    with open(out, 'w') as fid:
        fid.write('\n'.join(list(map(str, paths))) + '\n')
    print("Wrote file list to files.txt")


def write_annotations(annotations: Mapping[str, str], name: str = 'label',
                      path: Union[PathOrStr, None] = None):
    """Write sorted annotations CSV.

    Args:
    -----
    annotations: mapping
        A mapping of the form {name: annotation}.
    name: str
        Name of the annotation.
    path: pathlike or str, optional
        Path to write CSV. If None, filename is name.csv
    """
    df = pd.DataFrame.from_dict(annotations, orient='index', columns=[name])
    df.index.name = 'name'
    print()
    print(f"Value counts for {name}:")
    print(df[name].value_counts().sort_index())
    path = path or f'{name}.csv'
    df.sort_index().to_csv(path, header=True, index=True)
    print(f"Wrote CSV to {path}")
