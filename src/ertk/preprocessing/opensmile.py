from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import opensmile
from omegaconf import MISSING

from ertk.config import ERTKConfig

from .base import AudioClipProcessor, FeatureExtractor


@dataclass
class OpenSMILEExtractorConfig(ERTKConfig):
    opensmile_config: str = MISSING
    levels: List[str] = field(default_factory=lambda: ["func"])
    opensmile_opts: Dict[str, str] = field(default_factory=dict)


class OpenSMILEExtractor(
    FeatureExtractor,
    AudioClipProcessor,
    fname="opensmile",
    config=OpenSMILEExtractorConfig,
):
    config: OpenSMILEExtractorConfig

    def __init__(self, config: OpenSMILEExtractorConfig) -> None:
        super().__init__(config)

        if not Path(config.opensmile_config).exists():
            config.opensmile_config = opensmile.FeatureSet[config.opensmile_config]

        self.levels = config.levels
        self.smiles = [
            opensmile.Smile(
                config.opensmile_config, level, options=dict(config.opensmile_opts)
            )
            for level in config.levels
        ]
        self.feature_names = []
        for smile in self.smiles:
            self.feature_names.extend(smile.feature_names)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        xs = [smile(x, kwargs.pop("sr")) for smile in self.smiles]
        # TODO: allow other window/padding correction options
        cut = min(x.shape[2] for x in xs)
        xs = [x[:, :, :cut] for x in xs]
        x = np.concatenate(xs, axis=1)
        # Get single segment and transpose axes so that frames are first
        return x[0].T

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    @property
    def is_sequence(self) -> bool:
        return "func" not in self.levels
