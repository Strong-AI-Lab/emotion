import warnings
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from ertk.config import ERTKConfig

from .base import AudioClipProcessor, FeatureExtractor


def spectrogram(
    audio: np.ndarray,
    sr: float,
    kind: str = "mel",
    pre_emphasis: float = 0,
    window_size: float = 0.025,
    window_shift: float = 0.01,
    n_fft: Optional[int] = None,
    n_mels: int = 128,
    htk_mel: bool = False,
    n_chroma: int = 12,
    clip_db: Optional[float] = None,
    fmin: float = 0,
    fmax: Optional[float] = 8000,
    power: int = 2,
    to_db: bool = True,
):
    """General purpose spectrogram pipeline. Calculates spectrogram with
    optional pre-emphasis, clipping, mel/chroma transform, power and
    conversion to dB.

    Parameters
    ----------
    audio: np.ndarray
        Audio data. This should be a 1D array, or a 2D array with shape
        (1, n_samples).
    sr: int
        Sample rate.
    kind: str
        The kind of spectrogram to calculate. "mel" calculates a mel
        spectrogram, whereas "stft" just uses linear frequency.
    pre_emphasis: float
        Amount of pre-emphasis to apply. Default is 0 (no pre-emphasis).
    window_size: float
        Size of window in seconds. Default is 0.025 (25 ms).
    window_shift: float
        Amount window moves each frame, in seconds. Default is 0.01 (10
        ms).
    n_fft: int, optional
        Number of FFT bins. Default is to use the next power of 2
        greater than `window_size * sr`.
    n_mels: int
        Number of mel bands.
    htk_mel: bool,
        Use HTK formula for mel calculation.
    n_chroma: int
        Number of chroma bands.
    clip_db: float, optional
        Whether to clip noise floor at a given level in dB below
        maximum. Default is `None` which does no clipping. This is only
        used if `to_db` is `True`.
    fmin: float
        Minimum frequency for mel bands. Default is 0.
    fmax: float, optional
        Maximum frequency for mel bands. Default is 8000.
    power: int
        Raise spectrogram magnitudes to this power. Default is 2 which
        is the usual power spectrogram.
    to_db: bool
        Whether to convert spectrogram to dB units. Default is `True`.
        Note that this is mainly only useful if `power == 2`.

    Returns
    -------
    melspec: np.ndarray
        An array of shape (n_frames, n_mels) containing mel spectrogram.
        If power=2 this is the power spectrogram, and if to_db is `True`
        then this is in dB units.
    """
    audio = np.squeeze(audio)
    if audio.ndim != 1:
        raise ValueError(f"Audio should be in mono format, got shape {audio.shape}")

    window_samples = int(window_size * sr)
    stride_samples = int(window_shift * sr)

    # Pre-emphasis
    if pre_emphasis > 0:
        audio = librosa.effects.preemphasis(audio, pre_emphasis)

    # Spectrogram
    if not n_fft:
        n_fft = 2 ** int(np.ceil(np.log2(window_samples)))
    warnings.simplefilter("ignore", UserWarning)
    spec = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=stride_samples,
        win_length=window_samples,
        center=True,
        window="hann",
        pad_mode="reflect",
    )
    spec = np.abs(spec) ** power
    if kind == "mel":
        spec = librosa.feature.melspectrogram(
            S=spec, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk_mel
        )
    elif kind == "chroma":
        spec = librosa.feature.chroma_stft(S=spec, n_chroma=n_chroma)
    warnings.simplefilter("default", UserWarning)

    # Optional dB calculation
    if to_db:
        spec = librosa.power_to_db(spec, ref=np.max, top_db=clip_db)

    return spec.T


@dataclass
class SpectrogramExtractorConfig(ERTKConfig):
    kind: str = "mel"
    pre_emphasis: float = 0
    window_size: float = 0.025
    window_shift: float = 0.01
    n_mels: int = 128
    htk_mel: bool = False
    n_fft: Optional[int] = None
    n_chroma: int = 12
    clip_db: Optional[float] = None
    fmin: float = 0
    fmax: Optional[float] = 8000
    power: int = 2
    to_db: bool = True


class SpectrogramExtractor(
    FeatureExtractor,
    AudioClipProcessor,
    fname="spectrogram",
    config=SpectrogramExtractorConfig,
):
    config: SpectrogramExtractorConfig

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return spectrogram(x, sr=kwargs.pop("sr"), **self.config)  # type: ignore

    @property
    def dim(self) -> int:
        return self.config.n_mels

    @property
    def is_sequence(self) -> bool:
        return True
