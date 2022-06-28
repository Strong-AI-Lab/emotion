from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    idx_slice = tuple(slice(None) for _ in range(axis))

    num_frames = (x.size(axis) - frame_size) // frame_shift + 1
    if num_frames <= 0:
        if not pad:
            raise ValueError(
                "The length of the sequence is shorter than frame_size, but pad=False, "
                "so no frames will be generated."
            )
        num_frames = 0
    remainder = (x.size(axis) - frame_size) % frame_shift
    if remainder != 0:
        num_frames += 1 if pad else 0

    frames = []
    for i in range(0, x.size(axis) - frame_size + 1, frame_shift):
        frames.append(x[idx_slice + (slice(i, i + frame_size),)])
    if remainder != 0 and pad:
        if num_frames == 1:
            left = x.size(axis)
        else:
            left = frame_size + remainder - frame_shift
        # x[..., -left:, ...]
        final = x[idx_slice + (slice(-left, None),)]
        padding = (0, 0) * axis + (frame_size - left, 0) + (0, 0) * (x.ndim - axis - 1)
        padding = tuple(reversed(padding))
        frames.append(F.pad(final, padding))
    # Insert new dim before axis with num_frames, and axis is replaced
    # with the size of each frame.
    new_shape = x.shape[:axis] + (num_frames, frame_size) + x.shape[axis + 1 :]
    out = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
    return torch.stack(frames, axis, out=out)


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


def add_gaussian(x: torch.Tensor, sigma: float = 1):
    """Add Gaussian noise to a tensor or batch of tensors.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor of any shape.
    sigma: float
        Standard deviation of Gaussian.

    Returns
    -------
    augmented: torch.Tensor
        Augmented tensor of the same shape as the `x`.
    """
    noise = sigma * torch.randn(x.size(), dtype=x.dtype, device=x.device)
    return x + noise
