"""Processor using HuggingFace models.

.. autosummary::
    :toctree:

    HuggingFaceExtractorConfig
    HuggingFaceExtractor
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import torch
from omegaconf import MISSING

from ertk.config import ERTKConfig
from ertk.utils.array import is_mono_audio

from ._base import AudioClipProcessor, FeatureExtractor

__all__ = ["HuggingFaceExtractorConfig", "HuggingFaceExtractor"]


class Task(Enum):
    """Task to perform."""

    CTC = "ctc"
    """CTC speech recognition."""
    W2V2 = "w2v2"
    """Wav2Vec2 speech recognition. Use if CTC does not work."""
    EMBEDDINGS = "embeddings"
    """Extract embeddings."""
    S2T = "s2t"
    """Speech2Text speech recognition. Use for speech2text models."""
    WHISPER = "whisper"
    """Whisper speech recognition. Use for Whisper models."""
    AUTO = "auto"
    """Automatically detect model and task."""


class Agg(Enum):
    """Aggregation method for extracting embeddings."""

    MEAN = "mean"
    MAX = "max"
    NONE = "none"


@dataclass
class HuggingFaceExtractorConfig(ERTKConfig):
    """HuggingFace feature extractor configuration."""

    model: str = MISSING
    """Model name or path."""
    device: str = "cuda"
    """Device to run model on."""
    task: Task = Task.CTC
    """Task to perform."""
    agg: Optional[Agg] = Agg.MEAN
    """Aggregation method for embeddings."""
    layer: Optional[str] = "context"
    """Layer to extract embeddings from."""
    max_input_len: int = 1500000
    """Maximum input length."""
    whisper_lang: Optional[str] = None
    """Language to use for Whisper models."""
    max_new_tokens: int = 448
    """Maximum number of generated tokens for speech2text and Whisper
    models.
    """


class HuggingFaceExtractor(
    FeatureExtractor,
    AudioClipProcessor,
    fname="huggingface",
    config=HuggingFaceExtractorConfig,
):
    """Processor using HuggingFace models."""

    config: HuggingFaceExtractorConfig

    def __init__(self, config: HuggingFaceExtractorConfig) -> None:
        from transformers import (
            AutoFeatureExtractor,
            AutoModel,
            AutoModelForCTC,
            AutoProcessor,
            Speech2Text2Processor,
            Speech2TextForConditionalGeneration,
            Wav2Vec2ForCTC,
            Wav2Vec2Processor,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        super().__init__(config)

        print(f"Loading model from {config.model}")
        if config.task == Task.CTC:
            self.model = AutoModelForCTC.from_pretrained(config.model)
            self.processor = AutoProcessor.from_pretrained(config.model)
        elif config.task == Task.W2V2:
            self.model = Wav2Vec2ForCTC.from_pretrained(config.model)
            self.processor = Wav2Vec2Processor.from_pretrained(config.model)
        elif config.task == Task.S2T:
            self.model = Speech2TextForConditionalGeneration.from_pretrained(
                config.model
            )
            self.processor = Speech2Text2Processor.from_pretrained(config.model)
        elif config.task == Task.WHISPER:
            self.model = WhisperForConditionalGeneration.from_pretrained(config.model)
            self.processor = WhisperProcessor.from_pretrained(config.model)
        elif config.task == Task.EMBEDDINGS:
            self.model = AutoModel.from_pretrained(config.model)
            # Wav2Vec2Processor requires a tokeniser, which we don't need and may not
            # be available, so just use feature extractor instead.
            self.processor = AutoFeatureExtractor.from_pretrained(config.model)
        else:  # config.task == Task.AUTO
            self.model = AutoModel.from_pretrained(config.model)
            self.processor = AutoProcessor.from_pretrained(config.model)

        self.model.to(config.device).eval()
        torch.set_grad_enabled(False)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x = np.squeeze(x)
        if not is_mono_audio(x):
            raise ValueError("Audio must be mono")
        processed = self.processor(
            x[: self.config.max_input_len],
            sampling_rate=kwargs.pop("sr"),
            return_tensors="pt",
        )
        try:
            wav = processed.input_values.to(self.config.device)
        except AttributeError:
            wav = processed.input_features.to(self.config.device)

        if self.config.task in [Task.CTC, Task.W2V2]:
            output = self.model(wav).logits.argmax(-1)
            output = self.processor.batch_decode(output)
            return np.array([output], dtype=object)
        elif self.config.task in [Task.S2T, Task.WHISPER]:
            forced_decoder_ids = None
            if (
                self.config.task == Task.WHISPER
                and self.config.whisper_lang is not None
            ):
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    task="transcribe", language=self.config.whisper_lang
                )
            output = self.model.generate(
                wav,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=self.config.max_new_tokens,
            )
            output = self.processor.batch_decode(output, skip_special_tokens=True)
            return np.array([output], dtype=object)

        # Embeddings
        if self.config.layer == "context":
            output = self.model(wav).last_hidden_state.squeeze()
        else:
            output = self.model(wav).extract_features.squeeze()
        if self.config.agg == Agg.MEAN:
            output = output.mean(0)
        elif self.config.agg == Agg.MAX:
            output = output.max(0)
        return output.cpu().numpy()

    def process_batch(
        self, batch: Union[Iterable[np.ndarray], np.ndarray], **kwargs
    ) -> list[np.ndarray]:
        processed = self.processor(
            [x.squeeze()[: self.config.max_input_len] for x in batch],
            sampling_rate=kwargs.pop("sr"),
            return_tensors="pt",
            padding="longest",
        )
        try:
            wav = processed.input_values.to(self.config.device)
        except AttributeError:
            wav = processed.input_features.to(self.config.device)

        if self.config.task in [Task.CTC, Task.W2V2]:
            output = self.model(wav).logits.argmax(-1)
            output = self.processor.batch_decode(output)
            return [np.array([x], dtype=object) for x in output]
        elif self.config.task in [Task.S2T, Task.WHISPER]:
            forced_decoder_ids = None
            if (
                self.config.task == Task.WHISPER
                and self.config.whisper_lang is not None
            ):
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    task="transcribe", language=self.config.whisper_lang
                )
            output = self.model.generate(
                wav,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=self.config.max_new_tokens,
            )
            output = self.processor.batch_decode(output, skip_special_tokens=True)
            return [np.array([x], dtype=object) for x in output]

        # Embeddings
        raise RuntimeError("Cannot use batching for extracting embeddings.")

    @property
    def dim(self) -> int:
        if self.config.task in [Task.CTC, Task.W2V2, Task.WHISPER, Task.S2T]:
            return 1
        config = self.model.config
        for attr in ["hidden_size", "hidden_dim"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError("Unknown dimensionality.")

    @property
    def is_sequence(self) -> bool:
        # Single text string
        return False

    @property
    def feature_names(self) -> list[str]:
        if self.config.task in [Task.CTC, Task.W2V2, Task.WHISPER, Task.S2T]:
            return ["text"]
        return [f"{self.config.model}_{i}" for i in range(self.dim)]
