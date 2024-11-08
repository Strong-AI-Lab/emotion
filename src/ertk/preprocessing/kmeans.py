"""Kmeans vector quantiser.

.. autosummary::
    :toctree:

    KMeansExtractorConfig
    KMeansExtractor
"""

from dataclasses import dataclass

import numpy as np
from omegaconf import MISSING

from ertk.config import ERTKConfig

from ._base import FeatureExtractor

__all__ = ["KMeansExtractorConfig", "KMeansExtractor"]


@dataclass
class KMeansExtractorConfig(ERTKConfig):
    """KMeans feature extractor configuration."""

    pickle: str = MISSING
    """Path to pickle file."""


class KMeansExtractor(FeatureExtractor, fname="kmeans", config=KMeansExtractorConfig):
    """KMeans vector quantiser."""

    config: KMeansExtractorConfig

    def __init__(self, config: KMeansExtractorConfig) -> None:
        super().__init__(config)

        import joblib

        self.kmeans = joblib.load(config.pickle)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return np.expand_dims(self.kmeans.predict(x), axis=1)

    @property
    def is_sequence(self) -> bool:
        return True

    @property
    def feature_names(self) -> list[str]:
        return ["cluster_id"]
