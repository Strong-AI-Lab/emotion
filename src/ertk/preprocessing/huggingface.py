from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from omegaconf import MISSING
from transformers import AutoModel, AutoModelForCTC, AutoProcessor

from ertk.config import ERTKConfig

from ._base import AudioClipProcessor, FeatureExtractor


class _Task(Enum):
    CTC = "ctc"
    EMBEDDINGS = "embeddings"


class _Agg(Enum):
    MEAN = "mean"
    MAX = "max"
    NONE = "none"


@dataclass
class HuggingFaceExtractorConfig(ERTKConfig):
    model: str = MISSING
    device: str = "cuda"
    task: _Task = _Task.CTC
    agg: Optional[_Agg] = _Agg.MEAN
    layer: Optional[str] = "context"
    max_input_len: int = 1500000


class HuggingFaceExtractor(
    FeatureExtractor,
    AudioClipProcessor,
    fname="huggingface",
    config=HuggingFaceExtractorConfig,
):
    config: HuggingFaceExtractorConfig

    def __init__(self, config: HuggingFaceExtractorConfig) -> None:
        super().__init__(config)

        print(f"Loading model from {config.model}")
        if config.task == _Task.CTC:
            self.model = AutoModelForCTC.from_pretrained(config.model)
        else:
            self.model = AutoModel.from_pretrained(config.model)
        self.processor = AutoProcessor.from_pretrained(config.model)

        self.model.to(config.device).eval()
        torch.set_grad_enabled(False)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        wav = self.processor(
            x.squeeze()[: self.config.max_input_len],
            sampling_rate=kwargs.pop("sr"),
            return_tensors="pt",
            padding="longest",
        ).input_values.to(self.config.device)
        if self.config.task == _Task.CTC:
            output = self.model(wav).logits.argmax(-1)
            output = self.processor.batch_decode(output)
            return np.array([output], dtype=object)
        # Embeddings
        if self.config.layer == "context":
            output = self.model(wav).last_hidden_state.squeeze()
        else:
            output = self.model(wav).extract_features.squeeze()
        if self.config.agg == _Agg.MEAN:
            output = output.mean(0)
        elif self.config.agg == _Agg.MAX:
            output = output.max(0)
        return output.cpu().numpy()

    def process_batch(
        self, batch: Union[Iterable[np.ndarray], np.ndarray], **kwargs
    ) -> List[np.ndarray]:
        wav = self.processor(
            [x.squeeze()[: self.config.max_input_len] for x in batch],
            sampling_rate=kwargs.pop("sr"),
            return_tensors="pt",
            padding="longest",
        ).input_values.to(self.config.device)
        if self.config.task == _Task.CTC:
            output = self.model(wav).logits.argmax(-1)
            output = self.processor.batch_decode(output)
            return [np.array([x], dtype=object) for x in output]
        # Embeddings
        raise RuntimeError("Cannot use batching for extracting embeddings.")

    @property
    def dim(self) -> int:
        if self.config.task == _Task.CTC:
            return 1
        config = self.model.config
        for attr in ["hidden_size", "hidden_dim"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError("Unknown dimensionality.")

    @property
    def is_sequence(self) -> bool:
        return self.config.task == _Task.CTC

    @property
    def feature_names(self) -> List[str]:
        if self.config.task == _Task.CTC:
            return ["text"]
        return [f"{self.config.model}_{i}" for i in range(self.dim)]
