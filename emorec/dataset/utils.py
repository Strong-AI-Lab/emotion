import json
import subprocess
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import netCDF4
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

FFMPEG_RESAMPLE_CMD = "ffmpeg -i '{}' -ar 16000 -sample_fmt s16 -y '{}'"


def parse_regression_annotations(filename: Union[PathLike, str]) \
        -> Dict[str, Dict[str, float]]:
    """Returns a dict of the form {'name': {'v1': v1, ...}}."""
    # Need index_col to be False or None due to
    # https://github.com/pandas-dev/pandas/issues/9435
    df = pd.read_csv(filename, index_col=False, dtype='str')
    df = df.set_index(df.columns[0])
    annotations = df.to_dict(orient='index')
    return annotations


def parse_classification_annotations(filename: Union[PathLike, str]) \
        -> Dict[str, str]:
    """Returns a dict of the form {'name': emotion}."""
    # Need index_col to be False or None due to
    # https://github.com/pandas-dev/pandas/issues/9435
    df = pd.read_csv(filename, index_col=False, dtype='str')
    df = df.set_index(df.columns[0])
    annotations = df.to_dict()[df.columns[0]]
    return annotations


def get_audio_paths(file: Union[PathLike, str]) -> Sequence[Path]:
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


def write_netcdf_dataset(path: Union[PathLike, str],
                         names: List[str],
                         features: np.ndarray,
                         slices: Optional[List[int]] = None,
                         corpus: str = '',
                         annotations: Optional[np.ndarray] = None,
                         annotation_path: Optional[Union[PathLike, str]] = None,  # noqa
                         annotation_type: str = 'classification'):
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
    names: list of str
        A list of instance names.
    features: ndarray
        A features matrix of shape (length, n_features).
    slices: list of int, optional
        The size of each slice along axis 0 of features. If there is one
        vector per instance, then this will be all 1's, otherwise will
        have the length of the sequence corresponding to each instance.
    annotations: np.ndarray, optional
        Annotations obtained elsewhere.
    annotation_path: pathlike or str, optional
        The path to an annotation file.
    annotation_type: str
        The type of annotations, one of {regression, classification}.
    """
    dataset = netCDF4.Dataset(path, 'w')
    dataset.createDimension('instance', len(names))
    dataset.createDimension('concat', features.shape[0])
    dataset.createDimension('features', features.shape[1])

    if slices is None:
        slices = [1] * len(names)
    _slices = dataset.createVariable('slices', int, ('instance',))
    _slices[:] = slices

    filename = dataset.createVariable('filename', str, ('instance',))
    filename[:] = np.array(names)

    if annotation_path is not None:
        if annotation_type == 'regression':
            annotations = parse_regression_annotations(annotation_path)
            keys = next(iter(annotations.values())).keys()
            for k in keys:
                var = dataset.createVariable(k, np.float32, ('instance',))
                var[:] = np.array([annotations[x][k] for x in names])
            dataset.setncattr_string(
                'annotation_vars', json.dumps([k for k in keys]))
        elif annotation_type == 'classification':
            annotations = parse_classification_annotations(annotation_path)
            label_nominal = dataset.createVariable('label_nominal', str,
                                                   ('instance',))
            label_nominal[:] = np.array([annotations[x] for x in names])
            dataset.setncattr_string('annotation_vars',
                                     json.dumps(['label_nominal']))
    elif annotations is not None:
        if annotation_type == 'regression':
            for k, arr in annotations:
                var = dataset.createVariable(k, np.float32, ('instance',))
                var[:] = arr
            dataset.setncattr_string(
                'annotation_vars', json.dumps([k for k in annotations]))
        elif annotation_type == 'classification':
            label_nominal = dataset.createVariable('label_nominal', str,
                                                   ('instance',))
            label_nominal[:] = annotations
            dataset.setncattr_string('annotation_vars',
                                     json.dumps(['label_nominal']))
    else:
        label_nominal = dataset.createVariable('label_nominal', str,
                                               ('instance',))
        label_nominal[:] = np.array(['unknown' for _ in range(len(names))])
        dataset.setncattr_string('annotation_vars',
                                 json.dumps(['label_nominal']))

    _features = dataset.createVariable('features', np.float32,
                                       ('concat', 'features'))
    _features[:, :] = features

    dataset.setncattr_string('feature_dims',
                             json.dumps(['concat', 'features']))
    dataset.setncattr_string('corpus', corpus)
    dataset.close()


def resample_audio(paths: Iterable[Path], dir: Union[PathLike, str]):
    """Resample given audio clips to 16 kHz 16-bit WAV, and place in
    direcotory given by `dir`.
    """
    paths = list(paths)
    if len(paths) == 0:
        raise FileNotFoundError("No audio files found.")

    dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)
    print("Resampling {} audio files to {}".format(len(paths), dir))

    print("Using command line:\n{}".format(FFMPEG_RESAMPLE_CMD))
    Parallel(n_jobs=-1, verbose=1)(
        delayed(subprocess.run)(
            FFMPEG_RESAMPLE_CMD.format(path, dir / (path.stem + '.wav')),
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        ) for path in paths
    )


def write_filelist(paths: Iterable[Path],
                   out: Union[PathLike, str] = 'files.txt'):
    """Write sorted file list."""
    paths = sorted(paths, key=lambda p: p.stem)
    with open(out, 'w') as fid:
        fid.write('\n'.join(list(map(str, paths))) + '\n')
    print("Wrote file list to files.txt")


def write_labels(labels: Mapping[str, str],
                 out: Union[PathLike, str] = 'labels.csv'):
    """Write sorted emotion labels CSV. `labels` is a mapping of the
    form {name: label}.
    """
    df = pd.DataFrame.from_dict(labels, orient='index', columns=['Emotion'])
    df.index.name = 'Name'
    print(df['Emotion'].value_counts().sort_index())
    df.sort_index().to_csv(out, header=True, index=True)
    print("Wrote CSV to labels.csv")
