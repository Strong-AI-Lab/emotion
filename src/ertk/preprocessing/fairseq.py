from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from omegaconf import MISSING

from ertk.config import ERTKConfig

from .base import AudioClipProcessor, FeatureExtractor


class _Aggregation(Enum):
    MEAN = "mean"
    MAX = "max"
    NONE = "none"


@dataclass
class FairseqExtractorConfig(ERTKConfig):
    model_type: str = MISSING
    checkpoint: str = MISSING
    layer: str = "context"
    aggregate: _Aggregation = _Aggregation.MEAN
    device: str = "cuda"


class FairseqExtractor(
    FeatureExtractor, AudioClipProcessor, fname="fairseq", config=FairseqExtractorConfig
):
    config: FairseqExtractorConfig

    def __init__(self, config: FairseqExtractorConfig) -> None:
        super().__init__(config)

        print(f"Loading model from {config.checkpoint}")
        [model], args, task = load_model_ensemble_and_task([config.checkpoint])
        model.to(device=config.device)
        model.eval()
        self.model = model
        self.args = args
        self.task = task
        torch.set_grad_enabled(False)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if kwargs.pop("sr") != 16000:
            raise ValueError("sample rate should be 16kHz")

        tensor = torch.tensor(x, device=self.config.device).unsqueeze(0)
        if hasattr(self.task, "normalize") and self.task.normalize:
            tensor = F.layer_norm(tensor, tensor.shape)
        if self.config.model_type == "wav2vec":
            z = self.model.feature_extractor(tensor)
            if self.model.vector_quantizer is not None:
                z, _ = self.model.vector_quantizer.forward_idx(z)
            feats = z
            if self.config.layer == "context":
                feats = self.model.feature_aggregator(feats)
        elif self.config.model_type == "wav2vec2":
            if self.config.layer == "context":
                c = self.model.extract_features(tensor, None)["x"]
                # Transpose to (batch, feats, steps)
                feats = c.transpose(-2, -1)
            else:
                feats = self.model.feature_extractor(tensor)
        else:  # HuBERT
            if self.config.layer == "context":
                c, _ = self.model.extract_features(tensor)
                # Transpose to (batch, feats, steps)
                feats = c.transpose(-2, -1)
            else:
                feats = self.model.feature_extractor(tensor)

        if self.config.aggregate == _Aggregation.NONE:
            return feats[0].transpose(1, 0).cpu().numpy()
        elif self.config.aggregate == _Aggregation.MEAN:
            return feats[0].mean(-1).cpu().numpy()
        elif self.config.aggregate == _Aggregation.MAX:
            return feats[0].max(-1).cpu().numpy()
        else:
            raise ValueError(f"aggregation type {self.config.aggregate} not supported")

    @property
    def dim(self) -> int:
        return 1024

    @property
    def is_sequence(self) -> bool:
        return self.config.aggregate == _Aggregation.NONE
