from dataclasses import dataclass

import omegaconf
from ertk.preprocessing.fairseq._fairseq import load_model
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

        model, args = load_model([config.checkpoint])
        self.wav2vec = model
        self.w2v_args = args
        self.final = nn.Linear(512, self.n_classes)

    def forward(self, x, **kwargs):
        x = x.squeeze(-1)
        x = self.wav2vec.feature_extractor(x)
        if self.wav2vec.vector_quantizer is not None:
            x, _ = self.wav2vec.vector_quantizer.forward_idx(x)
        x = x.mean(-1)
        x = self.final(x)
        return x
