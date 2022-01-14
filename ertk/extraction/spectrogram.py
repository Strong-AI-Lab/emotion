import warnings
from typing import Optional

import librosa
import numpy as np


def spectrogram(
    audio: np.ndarray,
    sr: int,
    kind: str = "mel",
    pre_emphasis: float = 0,
    window_size: float = 0.025,
    window_shift: float = 0.01,
    n_mels: int = 128,
    clip_db: Optional[float] = None,
    fmin: float = 0,
    fmax: float = 8000,
    power: int = 2,
    to_db: bool = True,
):
    """General purpose spectrogram pipeline. Calculates spectrogram with
    optional pre-emphasis, clipping, mel transform, power and conversion
    to dB.

    Parameters:
    -----------
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
    n_mels: int
        Number of mel bands.
    clip_db: float, optional
        Whether to clip noise floor at a given level in dB below
        maximum. Default is `None` which does no clipping. This is only
        used if `to_db` is `True`.
    fmin: float
        Minimum frequency for mel bands. Default is 0.
    fmax: float
        Maximum frequency for mel bands. Default is 8000.
    power: int
        Raise spectrogram magnitudes to this power. Default is 2 which
        is the usual power spectrogram.
    to_db: bool
        Whether to convert spectrogram to dB units. Default is `True`.
        Note that this is mainly only useful if `power == 2`.

    Returns:
    --------
    melspec: np.ndarray
        An array of shape (n_frames, n_mels) containing mel spectrogram.
        If power=2 this is the power spectrogram, and if to_db is `True`
        then this is in dB units.
    """
    audio = np.squeeze(audio)
    if audio.ndim != 1:
        raise ValueError("Audio should be in mono format.")

    window_samples = int(window_size * sr)
    stride_samples = int(window_shift * sr)

    # Pre-emphasis
    if pre_emphasis > 0:
        audio = librosa.effects.preemphasis(audio, pre_emphasis)

    # Spectrogram
    warnings.simplefilter("ignore", UserWarning)
    n_fft = 2 ** int(np.ceil(np.log2(window_samples)))
    if kind == "mel":
        spec = librosa.feature.melspectrogram(
            audio,
            n_fft=n_fft,
            hop_length=stride_samples,
            win_length=window_samples,
            n_mels=n_mels,
            sr=sr,
            fmin=fmin,
            fmax=fmax,
            power=power,
        )
    else:
        spec = librosa.core.stft(
            audio,
            n_fft=n_fft,
            hop_length=stride_samples,
            win_length=window_samples,
        )
        spec = np.abs(spec) ** power
    warnings.simplefilter("default", UserWarning)

    # Optional dB calculation
    if to_db:
        spec = librosa.core.power_to_db(spec, ref=np.max, top_db=clip_db)

    return spec.T
