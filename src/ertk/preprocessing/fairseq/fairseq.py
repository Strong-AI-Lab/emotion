"""Fairseq processor."""

import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import joblib
import numpy as np
import torch
from omegaconf import MISSING

from ertk.config import ERTKConfig
from ertk.utils import PathOrStr, is_mono_audio

from .._base import AudioClipProcessor, FeatureExtractor

__all__ = ["FairseqExtractorConfig", "FairseqExtractor"]


class Agg(Enum):
    """Aggregation method."""

    MEAN = "mean"
    """Mean of the output sequence."""
    MAX = "max"
    """Max of the output sequence."""
    NONE = "none"
    """Return entire sequence."""


@dataclass
class FairseqExtractorConfig(ERTKConfig):
    """Fairseq feature extractor configuration."""

    checkpoint: str = MISSING
    """Path to model checkpoint."""
    layer: str = "context"
    """Layer to extract features from."""
    aggregate: Agg = Agg.MEAN
    """Aggregation method."""
    device: str = "cuda"
    """Device to run model on."""
    vq_path: Optional[str] = None
    """Path to vector quantiser."""
    vq_ids: bool = False
    """Whether to return VQ cluster ids."""
    vq_ids_as_string: bool = True
    """Whether to return VQ cluster ids as a single string of integers
    separated by spaces.
    """
    max_input_len: int = 1500000
    """Maximum input length."""


class FairseqExtractor(
    FeatureExtractor, AudioClipProcessor, fname="fairseq", config=FairseqExtractorConfig
):
    """Fairseq processor."""

    config: FairseqExtractorConfig

    def __init__(self, config: FairseqExtractorConfig) -> None:
        super().__init__(config)

        from . import _fairseq

        sys.modules["fairseq"] = _fairseq

        print(f"Loading model from {config.checkpoint}")
        model, _ = _fairseq.load_model(config.checkpoint)
        model.to(device=config.device)
        model.eval()
        self.model = model
        torch.set_grad_enabled(False)

        self.vq = None
        if config.vq_path:
            self.vq = joblib.load(config.vq_path)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if kwargs.pop("sr") != 16000:
            raise ValueError("Sample rate should be 16kHz")

        x = np.squeeze(x)
        if not is_mono_audio(x):
            raise ValueError("Audio must be mono")
        x = x[: self.config.max_input_len]
        tensor = torch.tensor(
            x, dtype=torch.float32, device=self.config.device
        ).unsqueeze(0)

        layer: Union[str, int]
        try:
            layer = int(self.config.layer)
        except ValueError:
            layer = self.config.layer
        if layer == "context":
            layer = -1

        # Embeddings
        feats = self.model.feature_extractor(tensor)
        if self.model.vector_quantizer is not None:
            feats, _ = self.model.vector_quantizer.forward_idx(feats)
        if layer == -1:
            feats = self.model.feature_aggregator(feats)

        if self.config.aggregate == Agg.NONE:
            feats = feats[0].transpose(1, 0).cpu().numpy()
        elif self.config.aggregate == Agg.MEAN:
            feats = feats[0].mean(-1).cpu().numpy()
        elif self.config.aggregate == Agg.MAX:
            feats = feats[0].max(-1).cpu().numpy()
        else:
            raise ValueError(f"Aggregation type {self.config.aggregate} not supported")

        if self.vq is None:
            return feats

        # Vector quantization
        if self.config.vq_ids:
            feats = self.vq.predict(feats)
            if self.config.vq_ids_as_string:
                feats = np.array([" ".join(map(str, feats))])
            feats = np.expand_dims(feats, -1)
        else:
            feats = self.vq.cluster_centers_[self.vq.predict(feats)]
        return feats

    def process_file(self, path: PathOrStr, sr: Optional[float] = None) -> np.ndarray:
        # Require 16 kHz sample rate
        return super().process_file(path, 16000)

    @property
    def dim(self) -> int:
        return 512

    @property
    def is_sequence(self) -> bool:
        return self.config.aggregate == Agg.NONE

    @property
    def feature_names(self) -> List[str]:
        if self.config.vq_ids:
            return ["wav2vec_vq_tok"]
        return [f"wav2vec_{i}" for i in range(self.dim)]
