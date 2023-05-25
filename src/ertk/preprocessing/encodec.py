from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
import torch

from ertk.config import ERTKConfig
from ertk.utils import is_mono_audio

from ._base import AudioClipProcessor, FeatureExtractor


class Agg(Enum):
    MEAN = "mean"
    MAX = "max"
    NONE = "none"


class Model(Enum):
    ENCODEC_24kHz = "24kHz"
    ENCODEC_48kHz = "48kHz"


@dataclass
class EncodecExtractorConfig(ERTKConfig):
    model: Model = Model.ENCODEC_48kHz
    aggregate: Agg = Agg.MEAN
    device: str = "cuda"
    vq_ids: bool = False
    vq_ids_as_string: bool = True
    max_input_len: int = 1500000


class EncodecExtractor(
    FeatureExtractor, AudioClipProcessor, fname="encodec", config=EncodecExtractorConfig
):
    config: EncodecExtractorConfig

    def __init__(self, config: EncodecExtractorConfig) -> None:
        super().__init__(config)

        from encodec import EncodecModel
        from encodec.utils import convert_audio

        self.convert_audio = convert_audio

        print(f"Loading {config.model.value} model")
        if config.model == Model.ENCODEC_48kHz:
            model = EncodecModel.encodec_model_48khz()
        else:
            model = EncodecModel.encodec_model_24khz()
        torch.set_grad_enabled(False)
        self.model = model.to(device=config.device).eval()

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        sr = kwargs.pop("sr")

        if not is_mono_audio(x):
            x = np.squeeze(x)
            if x.ndim > 1:
                raise ValueError("Audio must be mono")
        x = x[: self.config.max_input_len]
        audio = torch.as_tensor(x).unsqueeze(0)
        audio = self.convert_audio(
            audio, sr, self.model.sample_rate, self.model.channels
        ).to(device=self.config.device)
        frames = self.model.encode(audio.unsqueeze(0))
        codes = torch.cat([x[0] for x in frames], 2)
        if self.config.vq_ids:
            # (steps, n_q)
            feats = codes[0].cpu().transpose(0, 1).numpy()
            if self.config.vq_ids_as_string:
                feats = np.array([" ".join(map(str, x)) for x in feats])
                feats = np.expand_dims(feats, -1)
            return feats

        clusters = self.model.quantizer.decode(codes).cpu()
        n_q, dim, n_frames = clusters.shape
        if self.config.aggregate == Agg.MEAN:
            clusters = clusters.mean(2).reshape(n_q * dim).numpy()
        elif self.config.aggregate == Agg.MAX:
            clusters = clusters.max(2)[0].reshape(n_q * dim).numpy()
        elif self.config.aggregate == Agg.NONE:
            clusters = clusters.reshape(n_q * dim, n_frames).transpose(0, 1).numpy()
        else:
            raise ValueError(f"Invalid aggregation {self.config.aggregate}")
        return clusters

    @property
    def dim(self) -> int:
        if self.config.vq_ids:
            if self.config.vq_ids_as_string:
                return 1
            return self.model.quantizer.n_q
        return self.model.quantizer.n_q * self.model.quantizer.dimension

    @property
    def is_sequence(self) -> bool:
        return self.config.vq_ids or self.config.aggregate == Agg.NONE

    @property
    def feature_names(self) -> List[str]:
        if self.config.vq_ids:
            if self.config.vq_ids_as_string:
                return ["encodec_vq_ids"]
            return [f"encodec_vq_{i}" for i in range(self.dim)]
        return [f"encodec_{i}" for i in range(self.dim)]
