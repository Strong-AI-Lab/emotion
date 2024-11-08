"""Utilities for working with audio datasets."""

import logging
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

from joblib import delayed

from ertk.utils import PathOrStr, TqdmParallel

__all__ = [
    "get_audio_paths",
    "resample_audio",
    "resample_rename_clips",
    "write_filelist",
]


def get_audio_paths(path: PathOrStr, absolute: bool = True) -> list[Path]:
    """Given a path to a dir or list of audio files, return a sequence
    of absolute paths to those files. Note that this method handles
    relative paths but returns absolute paths, and doesn't resolve
    canonical paths (i.e. it doesn't follow symlinks.)

    Parameters
    ----------
    file: pathlike or str
        Path to a directory containing audio clips, or a file containing
        a list of paths to audio clips.
    absolute: bool
        If `True`, return absolute paths, otherwise return the paths
        already in the file/directory. Default is `True`.

    Returns
    -------
    list of paths to audio files. Paths will be absolute paths if
    `absolute=True`.
    """
    path = Path(path)
    if absolute:
        path = path.absolute()
    if path.is_dir():
        paths = [x for x in path.glob("*") if not x.is_dir()]
    else:
        par_dir = path.parent
        with open(path) as fid:
            paths = [par_dir / x.strip() for x in fid]
    return paths


def resample_rename_clips(
    mapping: Mapping[Path, Path], sr: int = 16000, *, n_jobs: int = -1
):
    """Resample given audio clips to 16 kHz 16-bit WAV.

    Parameters
    ----------
    mapping: mapping
        Mapping from source files to destination files.
    sr: int
        Sample rate.
    """
    dst_dirs = {x.parent for x in mapping.values()}
    for dir in dst_dirs:
        dir.mkdir(exist_ok=True, parents=True)

    opts = [
        "-nostdin",
        "-ar",
        f"{sr:d}",
        "-threads",
        "1" if n_jobs == -1 else "0",
        "-sample_fmt",
        "s16",
        "-ac",
        "1",
        "-y",
    ]
    logging.info(f"Resampling {len(mapping)} audio files")
    logging.info(f"Using FFmpeg options: {' '.join(opts)}")
    TqdmParallel(
        desc="Resampling audio", total=len(mapping), unit="file", n_jobs=n_jobs
    )(
        delayed(subprocess.run)(
            ["ffmpeg", "-i", str(src), *opts, str(dst)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        for src, dst in mapping.items()
    )


def resample_audio(
    paths: Iterable[PathOrStr], dir: PathOrStr, sr: int = 16000, *, n_jobs: int = -1
):
    """Resample given audio clips to 16 kHz 16-bit WAV, and place in
    direcotory given by `dir`.

    Parameters
    ----------
    paths: iterable of Path
        A collection of paths to audio files to resample.
    dir: Pathlike or str
        Output directory.
    sr: int
        Sample rate.
    """
    paths = list(map(Path, paths))
    if len(paths) == 0:
        raise FileNotFoundError("No audio files found.")

    resample_rename_clips(
        {x: Path(dir, f"{x.stem}.wav") for x in paths}, sr=sr, n_jobs=n_jobs
    )


def write_filelist(paths: Iterable[PathOrStr], path: PathOrStr):
    """Write sorted file list.

    Parameters
    ----------
    paths: iterable of Path
        Paths to audio clips.
    path: str
        Path of resulting file list.
    """
    paths = sorted(map(Path, paths), key=lambda p: p.stem)
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".txt")
    with open(path, "w") as fid:
        fid.write("\n".join(list(map(str, paths))) + "\n")
