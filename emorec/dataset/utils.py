import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from joblib import delayed

from ..utils import PathOrStr, TqdmParallel


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
        description="Resampling audio", total=len(paths), unit="file", n_jobs=-1
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
    print("Wrote file list to files.txt")
