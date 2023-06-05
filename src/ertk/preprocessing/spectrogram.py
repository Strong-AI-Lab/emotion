"""Spectrogram extraction."""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import librosa
import numpy as np

from ertk.config import ERTKConfig

from ._base import AudioClipProcessor, FeatureExtractor

__all__ = ["SpectrogramExtractorConfig", "SpectrogramExtractor", "spectrogram"]


def spectrogram(
    audio: np.ndarray,
    sr: float,
    kind: str = "mel",
    pre_emphasis: float = 0,
    window_size: float = 0.025,
    window_shift: float = 0.01,
    win_length_samp: Optional[int] = None,
    hop_length_samp: Optional[int] = None,
    n_fft: int = 2048,
    n_mels: int = 128,
    htk_mel: bool = False,
    n_chroma: int = 12,
    clip_db: Optional[float] = None,
    fmin: float = 0,
    fmax: Optional[float] = 8000,
    power: int = 2,
    to_log: Optional[str] = "db",
    mel_norm: str = "slaney",
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
    win_length_samp: int, optional
        The window length in samples. This overrides `window_size` if
        given.
    hop_length_samp: int, optional
        The hop size in samples. This overrides `window_shift` if given.
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
    to_log: str
        Whether to convert spectrogram to logarithmic domain. Default is
        `db` which converts to dB units. If `to_log=='log'` then the
        natual logarithm is taken. Note that this argument is mainly
        only useful if `power == 2`.
    mel_norm: str
        Normalisation to apply to mel filters. Default is "slaney" as in
        librosa.

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

    win_length = win_length_samp if win_length_samp else int(window_size * sr)
    hop_length = hop_length_samp if hop_length_samp else int(window_shift * sr)

    # Pre-emphasis
    if pre_emphasis > 0:
        audio = librosa.effects.preemphasis(audio, pre_emphasis)

    # Spectrogram
    warnings.simplefilter("ignore", UserWarning)
    spec = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        window="hann",
        pad_mode="reflect",
    )
    spec = np.abs(spec) ** power
    if kind == "mel":
        try:
            mel_norm = int(mel_norm)
        except ValueError:
            pass
        spec = librosa.feature.melspectrogram(
            S=spec,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk_mel,
            norm=mel_norm,
        )
    elif kind == "chroma":
        spec = librosa.feature.chroma_stft(S=spec, n_chroma=n_chroma)
    warnings.simplefilter("default", UserWarning)

    # Optional dB calculation
    if to_log == "db":
        spec = librosa.power_to_db(spec, ref=np.max, top_db=clip_db)
    elif to_log == "log":
        spec = np.log(np.maximum(spec, 1e-8))

    return spec.T


@dataclass
class SpectrogramExtractorConfig(ERTKConfig):
    """Configuration for SpectrogramExtractor."""

    kind: str = "mel"
    """The kind of spectrogram to calculate. "mel" calculates a mel
    spectrogram, whereas "stft" just uses linear frequency.
    """
    pre_emphasis: float = 0
    """Amount of pre-emphasis to apply. Default is 0 (no pre-emphasis)."""
    window_size: float = 0.025
    """Size of window in seconds. Default is 0.025 (25 ms)."""
    window_shift: float = 0.01
    """Amount window moves each frame, in seconds. Default is 0.01 (10 ms)."""
    win_length_samp: Optional[int] = None
    """The window length in samples. This overrides `window_size` if
    given.
    """
    hop_length_samp: Optional[int] = None
    """The hop size in samples. This overrides `window_shift` if given."""
    n_mels: int = 128
    """Number of mel bands."""
    htk_mel: bool = False
    """Use HTK formula for mel calculation."""
    n_fft: int = 2048
    """Number of FFT bins. Default is to use the next power of 2
    greater than `window_size * sr`.
    """
    n_chroma: int = 12
    """Number of chroma bands."""
    clip_db: Optional[float] = None
    """Whether to clip noise floor at a given level in dB below
    maximum. Default is `None` which does no clipping. This is only
    used if `to_db` is `True`.
    """
    fmin: float = 0
    """Minimum frequency for mel bands. Default is 0."""
    fmax: Optional[float] = 8000
    """Maximum frequency for mel bands. Default is 8000."""
    power: int = 2
    """Raise spectrogram magnitudes to this power. Default is 2 which
    is the usual power spectrogram.
    """
    to_log: Optional[str] = "db"
    """Whether to convert spectrogram to logarithmic domain. Default is
    `db` which converts to dB units. If `to_log=='log'` then the
    natual logarithm is taken. Note that this argument is mainly
    only useful if `power == 2`.
    """
    mel_norm: str = "slaney"
    """Normalisation to apply to mel filters. Default is "slaney" as in
    librosa.
    """


class SpectrogramExtractor(
    FeatureExtractor,
    AudioClipProcessor,
    fname="spectrogram",
    config=SpectrogramExtractorConfig,
):
    """Extracts a spectrogram from an audio clip."""

    config: SpectrogramExtractorConfig

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        config = SpectrogramExtractorConfig.to_dictconfig(self.config)
        return spectrogram(x, sr=kwargs.pop("sr"), **config)

    @property
    def dim(self) -> int:
        if self.config.kind == "mel":
            return self.config.n_mels
        elif self.config.kind == "chroma":
            return self.config.n_chroma
        else:
            return 1 + self.config.n_fft // 2

    @property
    def is_sequence(self) -> bool:
        return True

    @property
    def feature_names(self) -> List[str]:
        return [f"{self.config.kind}_{i}" for i in range(self.dim)]
