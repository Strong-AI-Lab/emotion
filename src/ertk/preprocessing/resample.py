"""Audio resampling using resampy."""

from dataclasses import dataclass
from typing import List

import numpy as np
import resampy

from ertk.config import ERTKConfig
from ertk.preprocessing._base import AudioClipProcessor

__all__ = ["ResampleConfig", "Resampler"]


@dataclass
class ResampleConfig(ERTKConfig):
    """Resample configuration."""

    sample_rate: int = 16000
    """Target sample rate."""
    filter: str = "kaiser_fast"
    """Resampling filter."""


class Resampler(AudioClipProcessor, fname="resample", config=ResampleConfig):
    """Audio resampler using resampy."""

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
