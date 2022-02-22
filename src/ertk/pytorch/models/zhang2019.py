import torch
import torch.nn.functional as F
from torch import nn

from .layers import Attention1D


def frame(
    x: torch.Tensor, frame_width: int, frame_step: int, pad: bool = False
) -> torch.Tensor:
    """Create a contiguous tensor of frames of x, where axis 1 of x is
    considered the time axis.

    Parameters
    ----------
    x: torch.Tensor
        A tensor of shape (B, T, ...) where B is the batch size, T is
        the length of the sequence.
    frame_width: int
        The width of each frame in steps.
    frame_step: int
        The number of steps each frame is shifted along the time axis.
    pad: bool
        Whether to pad the time axis or not, if T cannot evenly fit all
        frames.

    Returns
    -------
    frames: torch.Tensor
        A tensor of shape (B, N, frame_width, ...) where N is the number
        of frames. It will have one more dimension than x, and the final
        D - 3 dimensions will be the same as for x.
        N = 1 + int(pad) + (T - frame_width) / frame_step
    """
    batch_size, steps, *_ = x.size()
    if steps < frame_width:
        raise ValueError("Length of input sequence too small.")
    if (steps % frame_step) != (frame_width % frame_step):
        if pad:
            right_pad = frame_step - (steps % frame_step)
            x = F.pad(x, (0, 0) * (x.ndim - 2) + (0, right_pad))
        else:
            x = x[:, steps - (steps % frame_step), ...]
    steps -= steps % frame_step

    n_frames = 1 + int(pad) + (steps - frame_width) // frame_step
    new_shape = torch.Size((batch_size, n_frames, frame_width) + x.size()[2:])
    frames = torch.empty(new_shape, device=x.device, requires_grad=False)
    for i in range(n_frames):
        start = i * frame_step
        frames[:, i, :, ...] = x[:, start : start + frame_width, ...]
    return frames


class Model(nn.Module):
    def __init__(self, n_classes: int, n_features: int = 1) -> None:
        super().__init__()

        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size=40, padding=20), nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(40, 40, kernel_size=40, padding=20), nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool1d(10)

        self.gru = nn.GRU(1280, 128, num_layers=2, batch_first=True)
        self.attention = Attention1D(128)
        self.predict = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor):
        # x has shape (batch, steps, 640)
        batch_size, steps, _ = x.size()
        x = self.dropout(x).view(batch_size * steps, 1, 640)

        # Padding is even so we need to cut the last value
        extraction = self.conv1(x)[:, :, :-1]
        extraction = self.maxpool1(extraction)
        extraction = self.conv2(extraction)[:, :, :-1]
        # (batch * steps, 40, 320) -> (batch * steps, 320, 40)
        extraction = extraction.transpose(1, 2)
        extraction = self.maxpool2(extraction)
        extraction = extraction.transpose(1, 2).view(batch_size, steps, 1280)

        h0 = torch.zeros(2, batch_size, 128, dtype=x.dtype, device=x.device)
        gru, _ = self.gru(extraction, h0)
        att = self.attention(gru)
        out = self.predict(att)
        return out

    def preprocess_input(self, x: torch.Tensor):
        return frame(x, 640, 160).squeeze(-1)
