from pathlib import Path

import librosa
import numpy as np

from ertk.dataset.utils import (
    get_audio_paths,
    resample_audio,
    resample_rename_clips,
    write_filelist,
)

from .constants import audio_dir, resample_dir, resample_paths, test_data_dir


def test_get_audio_paths_dir() -> None:
    paths = get_audio_paths(audio_dir)
    assert len(paths) == 12


def test_get_audio_paths_file_1() -> None:
    paths = get_audio_paths(test_data_dir / "all_clips.txt")
    assert len(paths) == 12


def test_get_audio_paths_file_2() -> None:
    paths = get_audio_paths(test_data_dir / "subset.txt")
    assert len(paths) == 6


# This one is redundant since test_data_dir is absolute due to __file__
def test_get_audio_paths_absolute() -> None:
    paths = get_audio_paths(audio_dir, absolute=True)
    for path in paths:
        assert path.is_absolute()
        assert path == audio_dir / path.name
    assert len(paths) == 12


def test_resample(tmp_path: Path) -> None:
    resample_audio(resample_paths, tmp_path)
    paths = list(tmp_path.glob("*.wav"))
    for path in paths:
        audio, sr = librosa.load(path, sr=None)
        assert sr == 16000
        ref, _ = librosa.load(resample_dir / path.name, sr=None)
        assert np.allclose(audio, ref)
    assert len(paths) == 12


def test_resample_rename(tmp_path: Path) -> None:
    mapping = {x: tmp_path / (x.stem + "_rename.wav") for x in resample_paths}
    resample_rename_clips(mapping)
    paths = list(tmp_path.glob("*.wav"))
    for path in paths:
        assert path.name.endswith("_rename.wav")
        _, sr = librosa.load(path, sr=None, duration=0)
        assert sr == 16000
    assert len(paths) == 12


def test_write_filelist_path(tmp_path: Path) -> None:
    paths = ["test_clip1.wav", "clip2.mp3", "clip3.ogg"]
    out_path = tmp_path / "test_filelist.txt"
    write_filelist([Path(x) for x in paths], out_path)
    with open(out_path) as fid:
        lines = list(map(str.strip, fid))
    assert lines == sorted(paths)
