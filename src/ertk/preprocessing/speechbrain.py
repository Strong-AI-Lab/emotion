from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import torch
from omegaconf import MISSING

from ertk.config import ERTKConfig
from ertk.utils import is_mono_audio

from ._base import AudioClipProcessor, FeatureExtractor


class Task(Enum):
    WHISPER = "whisper"
    EMBEDDINGS = "embeddings"
    ASR = "asr"


class Agg(Enum):
    MEAN = "mean"
    MAX = "max"
    NONE = "none"


@dataclass
class SpeechBrainExtractorConfig(ERTKConfig):
    model: str = MISSING
    device: str = "cuda"
    task: Task = Task.EMBEDDINGS
    agg: Optional[Agg] = Agg.MEAN
    max_input_len: int = 1500000
    normalise: bool = True


class SpeechBrainExtractor(
    FeatureExtractor,
    AudioClipProcessor,
    fname="speechbrain",
    config=SpeechBrainExtractorConfig,
):
    config: SpeechBrainExtractorConfig

    def __init__(self, config: SpeechBrainExtractorConfig) -> None:
        from speechbrain.pretrained import (
            EncoderClassifier,
            EncoderDecoderASR,
            WhisperASR,
        )

        super().__init__(config)

        print(f"Loading model from {config.model}")
        if config.task == Task.EMBEDDINGS:
            self.model = EncoderClassifier.from_hparams(
                config.model, run_opts={"device": config.device}
            )
        elif config.task == Task.WHISPER:
            self.model = WhisperASR.from_hparams(
                config.model, run_opts={"device": config.device}
            )
        elif config.task == Task.ASR:
            self.model = EncoderDecoderASR.from_hparams(
                config.model, run_opts={"device": config.device}
            )

        self.model.to(config.device).eval()
        torch.set_grad_enabled(False)
        if config.task == Task.EMBEDDINGS:
            wav = torch.zeros(1, 16000, device=config.device)
            self._dim = self.model.encode_batch(wav).shape[2]

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x = np.squeeze(x)
        if not is_mono_audio(x):
            raise ValueError("Audio must be mono")
        wav = torch.from_numpy(x).to(self.config.device)
        wav = wav[: self.config.max_input_len].unsqueeze(0)
        assert kwargs.pop("sr") == 16000
        if self.config.task in [Task.WHISPER, Task.ASR]:
            text, _ = self.model.transcribe_batch(wav, torch.tensor([1.0]))
            return np.array([" ".join(text[0])], dtype=object)
        # Embeddings
        output = self.model.encode_batch(wav, normalize=self.config.normalise)
        output = output.squeeze(0)
        if self.config.agg == Agg.MEAN:
            output = output.mean(0)
        elif self.config.agg == Agg.MAX:
            output = output.max(0)
        return output.cpu().numpy()

    @property
    def dim(self) -> int:
        if self.config.task in [Task.WHISPER, Task.ASR]:
            return 1
        return self._dim

    @property
    def is_sequence(self) -> bool:
        # Single text string
        return False

    @property
    def feature_names(self) -> List[str]:
        if self.config.task in [Task.WHISPER, Task.ASR]:
            return ["text"]
        return [f"{self.config.model}_{i}" for i in range(self.dim)]
