from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from omegaconf import MISSING, OmegaConf

from ertk.config import ERTKConfig
from ertk.utils import PathOrStr

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
    arg_overrides: Dict[str, Any] = field(default_factory=dict)


class FairseqExtractor(
    FeatureExtractor, AudioClipProcessor, fname="fairseq", config=FairseqExtractorConfig
):
    config: FairseqExtractorConfig

    def __init__(self, config: FairseqExtractorConfig) -> None:
        super().__init__(config)

        print(f"Loading model from {config.checkpoint}")
        if self.config.model_type == "data2vec":
            # Have to do everything manually since the released model is
            # 'experimental'
            import fairseq.examples.data2vec.models.data2vec_audio  # noqa
            from fairseq import models, tasks

            state = torch.load(config.checkpoint)
            cfg = OmegaConf.create(state["cfg"])
            task = tasks.setup_task(cfg.task)

            _model_dataclass = models.MODEL_DATACLASS_REGISTRY["data2vec_audio"]
            for key in set(cfg.model):
                if key not in _model_dataclass.__dataclass_fields__:
                    del cfg.model[key]
            model = task.build_model(cfg.model)

            del state["model"]["_ema"]
            state["model"]["final_proj.weight"] = state["model"]["final_proj.0.weight"]
            del state["model"]["final_proj.0.weight"]
            state["model"]["final_proj.bias"] = state["model"]["final_proj.0.bias"]
            del state["model"]["final_proj.0.bias"]
            model.load_state_dict(state["model"])

            args = OmegaConf.create(state["args"])
        else:
            [model], args, task = load_model_ensemble_and_task(
                [config.checkpoint], arg_overrides=config.arg_overrides
            )
        model.to(device=config.device)
        model.eval()
        self.model = model
        self.args = args
        self.task = task
        torch.set_grad_enabled(False)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if kwargs.pop("sr") != 16000:
            raise ValueError("Sample rate should be 16kHz")

        tensor = torch.tensor(x, device=self.config.device).unsqueeze(0)
        if hasattr(self.task, "normalize") and self.task.normalize:
            tensor = F.layer_norm(tensor, tensor.shape)
        if self.config.model_type == "wav2vec":
            feats = self.model.feature_extractor(tensor)
            if self.model.vector_quantizer is not None:
                feats, _ = self.model.vector_quantizer.forward_idx(feats)
            if self.config.layer == "context":
                feats = self.model.feature_aggregator(feats)
        elif self.config.model_type in ["wav2vec2", "data2vec"]:
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
            raise ValueError(f"Aggregation type {self.config.aggregate} not supported")

    def process_file(self, path: PathOrStr, sr: Optional[float] = None) -> np.ndarray:
        # Require 16 kHz sample rate
        return super().process_file(path, 16000)

    @property
    def dim(self) -> int:
        return 1024

    @property
    def is_sequence(self) -> bool:
        return self.config.aggregate == _Aggregation.NONE
