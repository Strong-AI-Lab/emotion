from dataclasses import dataclass

import omegaconf
import torch
import torch.nn.functional as F
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from torch import nn

from ._base import PyTorchModelConfig, SimpleClassificationModel


@dataclass
class Wav2VecFineTuneConfig(PyTorchModelConfig):
    checkpoint: str = omegaconf.MISSING
    frozen_layers: int = -1


class Model(
    SimpleClassificationModel, fname="wav2vec_ft", config=Wav2VecFineTuneConfig
):
    def __init__(
        self,
        config: Wav2VecFineTuneConfig,
    ) -> None:
        super().__init__(config)

        [model], args, task = load_model_ensemble_and_task([config.checkpoint])
        self.wav2vec = model
        self.w2v_args = args
        # self.w2v_task = task

        # Needed for multi-GPU training because of
        # https://github.com/pytorch/fairseq/issues/3482
        self.w2v_cfg = dict(task.cfg)
        if "text_compression_level" in self.w2v_cfg:
            self.w2v_cfg["text_compression_level"] = str(
                self.w2v_cfg["text_compression_level"]
            )

        self.final = nn.Linear(512, self.n_classes)

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.wav2vec.feature_extractor(x)
        if self.wav2vec.vector_quantizer is not None:
            x, _ = self.wav2vec.vector_quantizer.forward_idx(x)
        x = x.mean(-1)
        x = self.final(x)
        return x

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.w2v_cfg["normalize"]:
            with torch.no_grad():
                x = F.layer_norm(x, x.shape)
        return x
