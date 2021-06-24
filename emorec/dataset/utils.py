import subprocess
from pathlib import Path
from typing import Iterable, List

from joblib import delayed

from ..utils import PathOrStr, TqdmParallel


def get_audio_paths(path: PathOrStr, absolute: bool = True) -> List[Path]:
    """Given a path to a dir or list of audio files, return a sequence
    of absolute paths to those files. Note that this method handles
    relative paths but returns absolute paths, and doesn't resolve
    canonical paths (i.e. it doesn't follow symlinks.)

    Args:
    -----
    file: pathlike or str
        Path to a directory containing audio clips, or a file containing
        a list of paths to audio clips.
    absolute: bool
        If `True`, return absolute paths, otherwise return the paths
        already in the file/directory. Default is `True`.

    Returns:
    --------
        List of paths to audio files. Paths will be absolute paths if
        `absolute=True`.
    """
    path = Path(path)
    if path.is_dir():
        paths = [x for x in path.glob("*") if not x.is_dir()]
    else:
        with open(path) as fid:
            paths = [path.parent / x.strip() for x in fid]
    if absolute:
        return [x.absolute() for x in paths]
    return paths


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

    opts = ["-nostdin", "-ar", "16000", "-sample_fmt", "s16", "-ac", "1", "-y"]
    print(f"Using FFmpeg options: {' '.join(opts)}")
    TqdmParallel(
        desc="Resampling audio", total=len(paths), unit="file", n_jobs=-1
    )(
        delayed(subprocess.run)(
            ["ffmpeg", "-i", str(path), *opts, str(dir / (path.stem + ".wav"))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        for path in paths
    )


def write_filelist(paths: Iterable[Path], out: PathOrStr = "files.txt"):
    """Write sorted file list."""
    paths = sorted(paths, key=lambda p: p.stem)
    with open(out, "w") as fid:
        fid.write("\n".join(list(map(str, paths))) + "\n")
    print(f"Wrote file list to {out}")
