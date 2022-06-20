from dataclasses import dataclass
from enum import Enum
from typing import List

import librosa
import numpy as np

from ertk.config import ERTKConfig
from ertk.preprocessing._base import AudioClipProcessor


class _VADMethod(Enum):
    mh2009 = "MH2009"
    librosa = "librosa"


@dataclass
class VADTrimmerConfig(ERTKConfig):
    method: _VADMethod = _VADMethod.librosa
    top_db: int = 60
    energy_thresh: float = 2.5
    freq_thresh: float = 150
    sf_thresh: float = 10
    window: float = 0.01
    debug: bool = False
    min_speech: int = 5
    min_silence: int = 10


def filter_silence_speech(x: np.ndarray, min_speech: int = 5, min_silence: int = 10):
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
    """Trim audio clips based on a VAD algorithm adapted from [1].

    [1] M. H. Moattar and M. M. Homayounpour, "A simple but efficient
    real-time Voice Activity Detection algorithm," 2009 17th European
    Signal Processing Conference, Glasgow, 2009, pp. 2549-2553.
    https://ieeexplore.ieee.org/abstract/document/7077834
    """
    window_samples = int(sr * window)
    frame_length = 1024
    sxx = librosa.stft(
        x,
        n_fft=frame_length,
        hop_length=window_samples,
        win_length=window_samples,
        center=True,
        window="hann",
        pad_mode="reflect",
    )
    sxx = np.abs(sxx)

    # e = np.log(sxx.mean(0))
    e = librosa.feature.rms(S=sxx, frame_length=frame_length)[0]
    min_e = e.min()

    freq = librosa.fft_frequencies(sr, frame_length)[1:]
    f = freq[sxx[1:, :].argmax(0)]
    min_f = f.min()

    sf = librosa.feature.spectral_flatness(x, S=sxx, power=2)[0]
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

    filter_silence_speech(speech, min_speech=min_speech, min_silence=min_silence)

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
    return x[start:end]


class VADTrimmer(AudioClipProcessor, fname="vad_trim", config=VADTrimmerConfig):
    config: VADTrimmerConfig

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        sr = kwargs.pop("sr")
        if self.config.method == _VADMethod.librosa:
            return librosa.effects.trim(x, top_db=self.config.top_db)[0]
        elif self.config.method == _VADMethod.mh2009:
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
