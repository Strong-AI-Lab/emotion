from dataclasses import dataclass
from typing import List

import numpy as np
import resampy

from ertk.config import ERTKConfig
from ertk.preprocessing._base import AudioClipProcessor


@dataclass
class ResampleConfig(ERTKConfig):
    sample_rate: int = 16000
    filter: str = "kaiser_fast"


class Resampler(AudioClipProcessor, fname="resample", config=ResampleConfig):
    config: ResampleConfig

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return resampy.resample(
            x,
            kwargs.pop("sr"),
            self.config.sample_rate,
            filter=self.config.filter,
            parallel=True,
        )

    @property
    def feature_names(self) -> List[str]:
        return ["pcm"]
