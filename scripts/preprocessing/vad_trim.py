"""Trim audio clips based on a VAD algorithm. One algorithm implemented
is adapted from [1]. The other uses the librosa.effects.trim() method.

[1] M. H. Moattar and M. M. Homayounpour, "A simple but efficient
real-time Voice Activity Detection algorithm," 2009 17th European Signal
Processing Conference, Glasgow, 2009, pp. 2549-2553.
https://ieeexplore.ieee.org/abstract/document/7077834
"""

from pathlib import Path

import click
import librosa
import numpy as np
import soundfile
from click_option_group import optgroup

from ertk.dataset import get_audio_paths
from ertk.utils import PathlibPath


def filter_silence_speech(
    speech: np.ndarray, min_speech: int = 5, min_silence: int = 10
):
    silence = True
    i = 0
    while i < len(speech):
        if silence and speech[i]:
            j = next(
                (j for j in range(i + 1, len(speech)) if not speech[j]), len(speech)
            )
            if j - i >= min_speech:
                silence = False
            else:
                speech[i:j] = False
        elif not silence and not speech[i]:
            j = next((j for j in range(i + 1, len(speech)) if speech[j]), len(speech))
            if j - i >= min_silence:
                silence = True
            else:
                speech[i:j] = True
        else:
            j = i + 1
        i = j


def mh2009_vad(
    path: Path,
    energy_thresh: float,
    freq_thresh: float,
    sf_thresh: float,
    window: float,
    debug: bool,
):
    audio, sr = librosa.load(path, sr=16000)
    print(path)
    window_samples = int(sr * window)
    s = librosa.stft(
        audio, n_fft=512, win_length=window_samples, hop_length=window_samples
    )
    sxx = np.abs(s) ** 2

    e = np.log(sxx.mean(0))
    min_e = e.min()

    freq = librosa.fft_frequencies(sr, 512)[1:]
    f = freq[sxx[1:, :].argmax(0)]
    min_f = f.min()

    sf = librosa.feature.spectral_flatness(audio, S=sxx, power=1)[0]
    sf = -10 * np.log10(sf / 10)
    min_sf = sf.min()

    thresh_e = energy_thresh
    thresh_f = freq_thresh
    thresh_sf = sf_thresh
    cond1 = (e - min_e >= thresh_e).astype(int)
    cond2 = (f - min_f >= thresh_f).astype(int)
    cond3 = (sf - min_sf >= thresh_sf).astype(int)
    count = cond1 + cond2 + cond3
    speech = count > 1

    filter_silence_speech(speech)

    if not np.any(speech):
        if debug:
            np.set_printoptions(precision=3, suppress=True)
            print(e - min_e)
            print(cond1)
            print(f - min_f)
            print(cond2)
            print(sf - min_sf)
            print(cond3)
        raise ValueError("No speech segments detected.")

    start = next(i for i in range(len(speech)) if speech[i])
    start *= window_samples
    end = next(i for i in range(len(speech), 0, -1) if speech[i - 1])
    end *= window_samples
    trimmed = audio[start:end]
    return trimmed


def librosa_vad(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=16000)
    trimmed, _ = librosa.effects.trim(audio)
    return trimmed


@click.command()
@click.argument("file", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("output", type=PathlibPath(file_okay=False))
@click.option(
    "--method",
    type=click.Choice(["mh2009", "librosa"], case_sensitive=False),
    default="MH2009",
    help="The method to use for VAD.",
)
@click.option("--debug", is_flag=True)
@optgroup.group('Arguments for method "mh2009"')
@optgroup.option(
    "--energy", "energy_thresh", type=click.FLOAT, default=2.5, help="Energy threshold"
)
@optgroup.option(
    "--freq",
    "freq_thresh",
    type=click.FLOAT,
    default=150,
    help="Max-frequency threshold",
)
@optgroup.option(
    "--sf",
    "sf_thresh",
    type=click.FLOAT,
    default=10,
    help="Spectral flatness threshold",
)
@optgroup.option("--window", type=click.FLOAT, default=0.01)
def main(
    file: Path,
    output: Path,
    energy_thresh: float,
    freq_thresh: float,
    sf_thresh: float,
    window: float,
    method: str,
    debug: bool,
):
    """Performs voice activity detection (VAD) on audio clips referred
    to by FILE and cuts out the beginning and end of the clips where
    there is no voice.
    """
    paths = get_audio_paths(file)
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if method == "mh2009":
            trimmed = mh2009_vad(
                path, energy_thresh, freq_thresh, sf_thresh, window, debug
            )
        else:
            trimmed = librosa_vad(path)

        output_path = output / path.name
        soundfile.write(output_path, trimmed, 16000)
        print(f"Wrote trimmed audio to {output_path}")


if __name__ == "__main__":
    main()
