"""OpenSMILE feature extraction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
from omegaconf import MISSING

from ertk.config import ERTKConfig
from ertk.utils import is_mono_audio

from ._base import AudioClipProcessor, FeatureExtractor

__all__ = ["OpenSMILEExtractorConfig", "OpenSMILEExtractor"]


@dataclass
class OpenSMILEExtractorConfig(ERTKConfig):
    """OpenSMILE feature extractor configuration."""

    opensmile_config: str = MISSING
    """OpenSMILE config file."""
    levels: List[str] = field(default_factory=lambda: ["func"])
    """OpenSMILE levels to extract."""
    opensmile_opts: Dict[str, str] = field(default_factory=dict)
    """Additional OpenSMILE options."""


class OpenSMILEExtractor(
    FeatureExtractor,
    AudioClipProcessor,
    fname="opensmile",
    config=OpenSMILEExtractorConfig,
):
    """OpenSMILE feature extractor."""

    config: OpenSMILEExtractorConfig

    def __init__(self, config: OpenSMILEExtractorConfig) -> None:
        import opensmile

        super().__init__(config)

        opensmile_config = config.opensmile_config
        if not Path(opensmile_config).exists():
            opensmile_config = opensmile.FeatureSet[opensmile_config]

        self.levels = config.levels
        self.smiles = [
            opensmile.Smile(
                opensmile_config, level, options=dict(config.opensmile_opts)
            )
            for level in config.levels
        ]
        self._feature_names = []
        for smile in self.smiles:
            self._feature_names.extend(smile.feature_names)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        sr = kwargs.pop("sr")
        x = np.squeeze(x)
        if not is_mono_audio(x):
            raise ValueError("Audio must be mono")
        xs = [smile(x, sr) for smile in self.smiles]
        # TODO: allow other window/padding correction options
        cut = min(x.shape[2] for x in xs)
        xs = [x[:, :, :cut] for x in xs]
        x = np.concatenate(xs, axis=1)
        # Get single segment and transpose axes so that frames are first
        return x[0].T

    @property
    def is_sequence(self) -> bool:
        return "func" not in self.levels

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
