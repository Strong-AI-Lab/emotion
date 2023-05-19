from dataclasses import dataclass
from typing import List

import numpy as np
from omegaconf import MISSING

from ertk.config import ERTKConfig

from ._base import FeatureExtractor


@dataclass
class KMeansExtractorConfig(ERTKConfig):
    pickle: str = MISSING


class KMeansExtractor(FeatureExtractor, fname="kmeans", config=KMeansExtractorConfig):
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
    def feature_names(self) -> List[str]:
        return ["cluster_id"]
