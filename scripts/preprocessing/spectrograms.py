import warnings
from pathlib import Path
from typing import Optional

import click
import joblib
import librosa
import numpy as np
from click.types import Choice
from click_option_group import RequiredAnyOptionGroup, optgroup
from matplotlib import pyplot as plt

from ertk.dataset import get_audio_paths, write_features
from ertk.utils import PathlibPath, TqdmParallel


def calculate_spectrogram(
    path: Path,
    channels: str = "mean",
    pre_emphasis: float = 0.95,
    skip: float = 0,
    length: Optional[float] = None,
    window_size: float = 0.025,
    window_shift: float = 0.01,
    n_mels: int = 240,
    clip: Optional[float] = None,
    fmin: float = 0,
    fmax: float = 8000,
):
    audio, sr = librosa.core.load(
        path, sr=None, mono=False, offset=skip, duration=length
    )
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, 0)

    window_samples = int(window_size * sr)
    stride_samples = int(window_shift * sr)

    # Channel fusion
    if channels == "left":
        audio = audio[0]
    elif channels == "right":
        audio = audio[1]
    elif channels == "mean":
        audio = np.mean(audio[:2], axis=0)
    elif channels == "diff":
        audio = audio[0] - audio[1]

    # Padding
    if length is not None and length > 0:
        length_samples = int(length * sr)
        if len(audio) < length_samples:
            audio = np.pad(audio, (0, length_samples - len(audio)))
        assert len(audio) == length_samples

    # Pre-emphasis
    if pre_emphasis > 0:
        audio = librosa.effects.preemphasis(audio, pre_emphasis)

    # Mel spectrogram
    warnings.simplefilter("ignore", UserWarning)
    n_fft = 2 ** int(np.ceil(np.log2(window_samples)))
    melspec = librosa.feature.melspectrogram(
        audio,
        n_mels=n_mels,
        sr=sr,
        n_fft=n_fft,
        hop_length=stride_samples,
        win_length=window_samples,
        fmin=fmin,
        fmax=fmax,
    )
    warnings.simplefilter("default", UserWarning)
    db_spectrogram = librosa.power_to_db(melspec, ref=np.max, top_db=clip)

    return db_spectrogram.T


@click.command()
@click.argument("corpus", type=str)
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@optgroup.group("Output format", cls=RequiredAnyOptionGroup)
@optgroup.option("--output", type=Path, help="Write dataset")
@optgroup.option("--preview", type=int, help="Preview spectrogram")
@optgroup.group("Spectrogram options")
@optgroup.option("--length", type=float, help="Optional max clip length")
@optgroup.option(
    "--skip", type=float, default=0, help="Optional amount to skip, in seconds"
)
@optgroup.option("--clip", type=float, help="Optional minimum power in dB")
@optgroup.option(
    "--window_size",
    type=float,
    default=0.025,
    show_default=True,
    help="Window size in seconds",
)
@optgroup.option(
    "--window_shift",
    type=float,
    default=0.010,
    show_default=True,
    help="Window shift in seconds",
)
@optgroup.option(
    "--mel_bands", type=int, default=240, show_default=True, help="Number of mel bands"
)
@optgroup.option(
    "--pre_emphasis",
    type=float,
    default=0.95,
    show_default=True,
    help="Pre-emphasis applied before processing",
)
@optgroup.option(
    "--channels",
    type=Choice(["left", "right", "mean", "diff"]),
    default="mean",
    show_default=True,
)
@optgroup.option(
    "--fmin", type=float, default=0, show_default=True, help="Min mel frequency"
)
@optgroup.option(
    "--fmax", type=float, default=8000, show_default=True, help="Max mel frequency"
)
def main(
    corpus: str,
    input: Path,
    output: Optional[Path],
    preview: Optional[int],
    length: Optional[float],
    skip: float,
    clip: Optional[float],
    window_size: float,
    window_shift: float,
    mel_bands: int,
    pre_emphasis: float,
    channels: str,
    fmin: float,
    fmax: float,
):
    """Extracts spectrograms from audio files listed in INPUT file and
    creates a netCFD4 dataset holding the data. CORPUS specifies the
    corpus.
    """

    paths = get_audio_paths(input)
    kwargs = dict(
        channels=channels,
        skip=skip,
        length=length,
        window_size=window_size,
        pre_emphasis=pre_emphasis,
        window_shift=window_shift,
        n_mels=mel_bands,
        clip=clip,
        fmin=fmin,
        fmax=fmax,
    )

    if preview is not None:
        idx = preview if preview > -1 else np.random.default_rng().integers(len(paths))
        spectrogram = calculate_spectrogram(paths[idx], **kwargs)
        plt.figure()
        plt.title(f"Spectrogram for {paths[idx]}.")
        plt.imshow(spectrogram)
        plt.show()
        return

    if not output:
        raise ValueError("Must specify either --preview or --output options.")

    specs = TqdmParallel(len(paths), "Generating spectrograms", n_jobs=-1)(
        joblib.delayed(calculate_spectrogram)(path, **kwargs) for path in paths
    )

    filenames = [x.stem for x in paths]
    if output:
        feature_names = [f"meldB{i + 1}" for i in range(mel_bands)]
        write_features(
            output,
            corpus=corpus,
            names=filenames,
            features=specs,
            feature_names=feature_names,
        )
        print(f"Wrote dataset to {output}")


if __name__ == "__main__":
    main()
