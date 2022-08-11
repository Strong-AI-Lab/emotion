import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
from phonemizer import phonemize
from phonemizer.separator import Separator

from ertk.config import ERTKConfig

from ._base import FeatureExtractor


@dataclass
class PhonemizeConfig(ERTKConfig):
    language: str = "en-us"
    backend: str = "festival"
    ph_sep: Optional[str] = " "
    syl_sep: Optional[str] = None
    word_sep: Optional[str] = None
    preserve_punctuation: bool = False


class Phonemizer(FeatureExtractor, fname="phonemize", config=PhonemizeConfig):
    config: PhonemizeConfig

    def __init__(self, config: PhonemizeConfig) -> None:
        super().__init__(config)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x = x.squeeze()
        if x.ndim > 0:
            text = " ".join(x)
        else:
            text = x.item()
        res = phonemize(
            text,
            language=self.config.language,
            backend=self.config.backend,
            separator=Separator(
                phone=self.config.ph_sep,
                syllable=self.config.syl_sep,
                word=self.config.word_sep,
            ),
            preserve_punctuation=self.config.preserve_punctuation,
        )
        return np.array([res])

    def process_batch(
        self, batch: Union[Iterable[np.ndarray], np.ndarray], **kwargs
    ) -> List[np.ndarray]:
        if isinstance(batch, np.ndarray):
            text = batch.squeeze().tolist()
        else:
            text = [x.squeeze().item() for x in batch]
        res = phonemize(
            text,
            language=self.config.language,
            backend=self.config.backend,
            separator=Separator(
                phone=self.config.ph_sep,
                syllable=self.config.syl_sep,
                word=self.config.word_sep,
            ),
            preserve_punctuation=self.config.preserve_punctuation,
            njobs=os.cpu_count(),
        )
        return [np.array([x]) for x in res]

    @property
    def dim(self) -> int:
        return 1

    @property
    def is_sequence(self) -> bool:
        # We output a single string of phonemes
        return False

    @property
    def feature_names(self) -> List[str]:
        return ["phones"]
