import torch
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()

        self.conv8 = nn.Sequential(
            nn.Conv1d(n_features, 384, kernel_size=8, padding=20), nn.ReLU(inplace=True)
        )
        self.conv16 = nn.Sequential(
            nn.Conv1d(n_features, 384, kernel_size=16, padding=20),
            nn.ReLU(inplace=True),
        )
        self.conv32 = nn.Sequential(
            nn.Conv1d(n_features, 384, kernel_size=32, padding=20),
            nn.ReLU(inplace=True),
        )
        self.conv64 = nn.Sequential(
            nn.Conv1d(n_features, 384, kernel_size=64, padding=20),
            nn.ReLU(inplace=True),
        )
        self.dense = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, steps, features) -> (batch, features, steps)
        x.transpose_(1, 2)
        steps = x.size(-1)
        c1 = F.max_pool1d(self.conv8(x), steps - 7)
        c2 = F.max_pool1d(self.conv16(x), steps - 15)
        c3 = F.max_pool1d(self.conv32(x), steps - 31)
        c4 = F.max_pool1d(self.conv64(x), steps - 63)
        x = torch.cat([c1, c2, c3, c4], -2).squeeze(-1)
        out = self.dense(x)
        return out
