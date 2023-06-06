"""Voice activity detection (VAD) trimming."""

from dataclasses import dataclass
from enum import Enum
from typing import List

import librosa
import numpy as np

from ertk.config import ERTKConfig

from ._base import AudioClipProcessor

__all__ = ["VADTrimmerConfig", "VADTrimmer"]


class Method(Enum):
    """VAD trimming method."""

    mh2009 = "MH2009"
    """Moattar and Homayounpour (2009)"""
    librosa = "librosa"
    """librosa.effects.trim"""


@dataclass
class VADTrimmerConfig(ERTKConfig):
    """VAD trimming configuration."""

    method: Method = Method.librosa
    """VAD trimming method."""
    top_db: int = 60
    """Top decibel threshold for librosa VAD."""
    energy_thresh: float = 2.5
    """Energy threshold for MH2009 VAD."""
    freq_thresh: float = 150
    """Frequency threshold for MH2009 VAD."""
    sf_thresh: float = 10
    """Spectral flatness threshold for MH2009 VAD."""
    window: float = 0.01
    """Window size in seconds for MH2009 VAD."""
    debug: bool = False
    """Whether to plot debug figures."""
    min_speech: int = 5
    """Minimum number of consecutive speech frames to keep."""
    min_silence: int = 10
    """Minimum number of consecutive silence frames to keep."""


def _filter_silence_speech(x: np.ndarray, min_speech: int = 5, min_silence: int = 10):
    """Smooths a boolean array indicating segments of speech based on
    the given speech and silence thresholds.
    """
    silence = True
    i = 0
    while i < len(x):
        if silence and x[i]:
            # Going from silence to speech
            j = next((j for j in range(i + 1, len(x)) if not x[j]), len(x))
            if j - i >= min_speech:
                silence = False
            else:
                x[i:j] = False
        elif not silence and not x[i]:
            # Going from speech to silence
            j = next((j for j in range(i + 1, len(x)) if x[j]), len(x))
            if j - i >= min_silence:
                silence = True
            else:
                x[i:j] = True
        else:
            j = i + 1
        i = j


def mh2009_vad(
    x: np.ndarray,
    sr: float,
    energy_thresh: float = 0.05,
    freq_thresh: float = 150,
    sf_thresh: float = 10,
    window: float = 0.01,
    debug: bool = False,
    min_speech: int = 5,
    min_silence: int = 10,
):
    """Trim audio clips based on a VAD algorithm adapted from [1]_.

    Parameters
    ----------
    x : np.ndarray
        Audio signal.
    sr : float
        Sample rate.
    energy_thresh : float, optional
        Energy threshold, by default 0.05.
    freq_thresh : float, optional
        Frequency threshold, by default 150.
    sf_thresh : float, optional
        Spectral flatness threshold, by default 10.
    window : float, optional
        Window size in seconds, by default 0.01.
    debug : bool, optional
        Whether to plot debug figures, by default False.
    min_speech : int, optional
        Minimum number of consecutive speech frames to keep, by default
        5.
    min_silence : int, optional
        Minimum number of consecutive silence frames to keep, by default
        10.

    Returns
    -------
    np.ndarray
        Boolean array indicating segments of voicing.

    References
    ----------
    .. [1] M. H. Moattar and M. M. Homayounpour, "A simple but efficient
           real-time Voice Activity Detection algorithm," 2009 17th
           European Signal Processing Conference, Glasgow, 2009, pp.
           2549-2553.
           https://ieeexplore.ieee.org/abstract/document/7077834
    """
    window_samples = int(sr * window)
    n_fft = 1024
    sxx = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=window_samples,
        win_length=window_samples,
        center=True,
        window="hann",
        pad_mode="reflect",
    )
    sxx = np.abs(sxx)

    # RMS energy
    e = librosa.feature.rms(S=sxx, frame_length=n_fft)[0]
    min_e = e.min()

    # Dominant frequency (excluding DC) in each frame
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[1:]
    f = freq[sxx[1:, :].argmax(0)]
    min_f = f.min()

    # Spectral flatness
    sf = librosa.feature.spectral_flatness(S=sxx, n_fft=n_fft, power=2)[0]
    sf_db = -10 * np.log10(sf / 10)
    min_sf = sf_db.min()

    cond1 = (e - min_e >= energy_thresh).astype(int)
    cond2 = (f - min_f >= freq_thresh).astype(int)
    cond3 = (sf_db - min_sf >= sf_thresh).astype(int)
    count = cond1 + cond2 + cond3
    speech = count > 1

    _filter_silence_speech(speech, min_speech=min_speech, min_silence=min_silence)

    if not np.any(speech):
        if debug:
            np.set_printoptions(precision=3, suppress=True)
            print(e - min_e)
            print(cond1)
            print(f - min_f)
            print(cond2)
            print(sf_db - min_sf)
            print(cond3)
        raise ValueError("No speech segments detected.")

    start = next(i for i in range(len(speech)) if speech[i])
    start *= window_samples
    end = next(i for i in range(len(speech), 0, -1) if speech[i - 1])
    end *= window_samples
    return x[start:end]


class VADTrimmer(AudioClipProcessor, fname="vad_trim", config=VADTrimmerConfig):
    """Voice activity detection (VAD) trimmer."""

    config: VADTrimmerConfig

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        sr = kwargs.pop("sr")
        if self.config.method == Method.librosa:
            return librosa.effects.trim(x, top_db=self.config.top_db)[0]
        elif self.config.method == Method.mh2009:
            return mh2009_vad(
                x,
                sr=sr,
                energy_thresh=self.config.energy_thresh,
                freq_thresh=self.config.freq_thresh,
                sf_thresh=self.config.sf_thresh,
                window=self.config.window,
                min_speech=self.config.min_speech,
                min_silence=self.config.min_silence,
            )
        raise NotImplementedError(f"Method {self.config.method} not implemented.")

    @property
    def feature_names(self) -> List[str]:
        return ["pcm"]
