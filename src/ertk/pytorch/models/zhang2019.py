import torch
from torch import nn

from ertk.pytorch.utils import frame_tensor

from ._base import PyTorchModelConfig, SimpleClassificationModel
from .layers import Attention1D


class Model(SimpleClassificationModel):
    def __init__(self, config: PyTorchModelConfig) -> None:
        super().__init__(config)

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
        self.predict = nn.Linear(128, config.n_classes)

    def forward(self, x: torch.Tensor):  # type: ignore
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
        return frame_tensor(
            x, frame_size=640, frame_shift=160, pad=False, axis=1
        ).squeeze(-1)
