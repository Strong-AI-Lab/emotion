from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def load_tensorboard():
    """Replaces the TensorFlow tensorboard with the tensorboard stub.
    This avoids issues with having both PyTorch and TensorFlow installed
    simultaneously.
    """
    import tensorboard.compat.tensorflow_stub.io.gfile as _gfile
    import tensorflow as _tensorflow

    _tensorflow.io.gfile = _gfile


@dataclass
class MTLTaskConfig:
    name: str
    weight: float
    output_dim: int
    loss: nn.Module


def frame_tensor(
    x: torch.Tensor,
    frame_size: int,
    frame_shift: int,
    pad: bool = False,
    axis: int = 0,
) -> torch.Tensor:
    """Frames an array over a given axis with optional padding and
    copying.

    Parameters
    ----------
    x: torch.Tensor
        Array to frame.
    frame_size: int
        Width of frame.
    frame_shift: int
        Amount each window is moved for subsequent frames.
    pad: bool
        If the frames don't neatly fit the axis length, this indicated
        whether to pad or cut the remaining length. Default is `False`
        which means any remaining length is cut. If `pad=True` then any
        remaining length is padded with zeros and an additional frame is
        produced. If frames neatly fit into the axis length then this
        option has no effect.
    axis: int
        Axis to frame over. Default is 0.

    Returns
    -------
    frames: torch.Tensor
        The resulting frames.
    """

    if axis < 0:
        axis += x.ndim

    L = x.size(axis)
    if L < frame_size and not pad:
        raise ValueError(
            "The length of the sequence is shorter than frame_size, but pad=False, "
            "so no frames will be generated."
        )

    num_frames = (L - frame_size) // frame_shift + 1
    remainder = (L - frame_size) % frame_shift
    if remainder != 0 and pad:
        if num_frames <= 0:
            left = L
        else:
            left = frame_size + remainder - frame_shift
        padding = (0, 0) * axis + (frame_size - left, 0) + (0, 0) * (x.ndim - axis - 1)
        x = F.pad(x, tuple(reversed(padding)))
    framed = x.unfold(axis, frame_size, frame_shift)
    if axis != framed.ndim - 2:
        return framed.transpose(axis + 1, framed.ndim - 1)
    return framed


def add_signal_awgn(x: torch.Tensor, snr: float = 20) -> torch.Tensor:
    """Add additive white Gaussian noise (AWGN) to a signal or batch of
    signals.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor of shape (signal_len,) or (batch_size, signal_len).
    snr: float
        Target signal-to-noise ratio.

    Returns
    -------
    augmented: torch.Tensor
        Augmented tensor of the same shape as the `x`.
    """
    x = x.squeeze(-1)
    noise = torch.randn(x.size(), dtype=x.dtype, device=x.device)
    x_power = torch.mean(x ** 2, -1)
    n_power = torch.mean(noise ** 2, -1)
    K = torch.sqrt(x_power / n_power * 10 ** (-snr / 10))
    return x + K.unsqueeze(-1) * noise


def add_gaussian(x: torch.Tensor, sigma: float = 1, multiplicative: bool = True):
    """Add Gaussian noise to a tensor or batch of tensors.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor of any shape.
    sigma: float
        Standard deviation of Gaussian.
    multiplicative: bool
        Additive or multiplicative noise.

    Returns
    -------
    augmented: torch.Tensor
        Augmented tensor of the same shape as the `x`.
    """
    noise = sigma * torch.randn(x.size(), dtype=x.dtype, device=x.device)
    if multiplicative:
        return x * (1 + noise)
    else:
        return x + noise


def spectrogram(
    x: torch.Tensor,
    sr: float,
    kind: str = "mel",
    pre_emphasis: float = 0,
    window_size: float = 0.025,
    window_shift: float = 0.01,
    n_fft: int = 2048,
    n_mels: int = 128,
    htk_mel: bool = False,
    n_chroma: int = 12,
    clip_db: Optional[float] = None,
    fmin: float = 0,
    fmax: Optional[float] = 8000,
    power: int = 2,
    to_db: bool = True,
):
    window_samples = int(window_size * sr)
    stride_samples = int(window_shift * sr)
    if kind == "mel":
        module = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=window_samples,
            hop_length=stride_samples,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            power=power,
            mel_scale="htk" if htk_mel else "slaney",
        )
    else:
        module = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=window_samples,
            hop_length=stride_samples,
            power=power,
        )
    with torch.no_grad():
        return module(x).transpose(-1, -2)
