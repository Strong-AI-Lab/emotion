"""Phonemize text using the phonemizer library."""

import os
import re
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from ertk.config import ERTKConfig

from ._base import FeatureExtractor

__all__ = ["PhonemizeConfig", "Phonemizer"]


@dataclass
class PhonemizeConfig(ERTKConfig):
    """Phonemizer configuration."""

    language: str = "en-us"
    """Language to use."""
    backend: str = "festival"
    """Backend to use."""
    ph_sep: Optional[str] = " "
    """Phoneme separator."""
    syl_sep: Optional[str] = None
    """Syllable separator."""
    word_sep: Optional[str] = None
    """Word separator."""
    preserve_punctuation: bool = False
    """Preserve punctuation."""
    preserve_empty_lines: bool = True
    """Preserve empty lines."""
    strip: bool = False
    """Strip whitespace."""
    language_switch: str = "keep-flags"
    """Language switch flagging. One of "keep-flags", "remove-flags",
    "remove-utterance".
    """


def _festival_punct(texts: List[str]) -> Tuple[List[str], int]:
    newtexts = []
    invalid = 0
    for x in texts:
        if re.match(r"^\W+$", x):
            newtexts.append("")
            invalid += 1
        else:
            newtexts.append(x)
    return newtexts, invalid


class Phonemizer(FeatureExtractor, fname="phonemize", config=PhonemizeConfig):
    """Phonemizer using the phonemizer library."""

    config: PhonemizeConfig

    def __init__(self, config: PhonemizeConfig) -> None:
        from phonemizer import phonemize
        from phonemizer.separator import Separator

        super().__init__(config)
        self.sep = Separator(
            phone=self.config.ph_sep,
            syllable=self.config.syl_sep,
            word=self.config.word_sep,
        )
        self.phonemize = phonemize

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x = x.squeeze()
        if x.ndim > 0:
            text = " ".join(x)
        else:
            text = x.item()
        if self.config.backend == "festival":
            corrected, invalid = _festival_punct([text])
            if invalid > 0:
                warnings.warn(
                    "Some text contains only punctuation, which causes issues with "
                    "the festival backend. Setting to empty string."
                )
                text = corrected[0]
        res = self.phonemize(
            text,
            language=self.config.language,
            backend=self.config.backend,
            separator=self.sep,
            preserve_punctuation=self.config.preserve_punctuation,
            preserve_empty_lines=self.config.preserve_empty_lines,
            strip=self.config.strip,
            language_switch=self.config.language_switch,
        )
        return np.array([res])

    def process_batch(
        self, batch: Union[Iterable[np.ndarray], np.ndarray], **kwargs
    ) -> List[np.ndarray]:
        if isinstance(batch, np.ndarray):
            text = batch.squeeze().tolist()
        else:
            text = [x.squeeze().item() for x in batch]
        if self.config.backend == "festival":
            corrected, invalid = _festival_punct(text)
            if invalid > 0:
                warnings.warn(
                    "Some text contains only punctuation, which causes issues with "
                    "the festival backend. Setting to empty string."
                )
                text = corrected
        res = self.phonemize(
            text,
            language=self.config.language,
            backend=self.config.backend,
            separator=self.sep,
            preserve_punctuation=self.config.preserve_punctuation,
            preserve_empty_lines=self.config.preserve_empty_lines,
            strip=self.config.strip,
            language_switch=self.config.language_switch,
            njobs=os.cpu_count(),
        )
        return [np.array([x]) for x in res]

    @property
    def is_sequence(self) -> bool:
        # We output a single string of phonemes
        return False

    @property
    def feature_names(self) -> List[str]:
        return ["phones"]
