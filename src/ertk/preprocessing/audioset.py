"""Audioset feature extractors.

.. autosummary::
    :toctree:

    YAMNetExtractor
    YAMNetExtractorConfig
    VGGishExtractor
    VGGishExtractorConfig
"""

from dataclasses import dataclass

import numpy as np

from ertk.config import ERTKConfig
from ertk.utils import is_mono_audio

from ._base import AudioClipProcessor, FeatureExtractor

__all__ = [
    "YAMNetExtractor",
    "YAMNetExtractorConfig",
    "VGGishExtractor",
    "VGGishExtractorConfig",
]


@dataclass
class YAMNetExtractorConfig(ERTKConfig):
    """Configuration for a Yamnet extractor."""


class YAMNetExtractor(
    FeatureExtractor, AudioClipProcessor, fname="yamnet", config=YAMNetExtractorConfig
):
    """A YAMNet extractor."""

    config: YAMNetExtractorConfig

    def __init__(self, config: YAMNetExtractorConfig) -> None:
        super().__init__(config)

        from ertk.tensorflow import init_gpu_memory_growth

        init_gpu_memory_growth()

        import tensorflow_hub as hub

        self.model = hub.load(
            "https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1"  # noqa
        )

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if not is_mono_audio(x):
            raise ValueError("YAMNet extractor requires mono audio input.")
        _, embeddings, _ = self.model(x)
        return np.mean(embeddings.numpy(), 0)

    @property
    def dim(self) -> int:
        return 1024

    @property
    def is_sequence(self) -> bool:
        return False

    @property
    def feature_names(self) -> list[str]:
        return [f"yamnet_{i}" for i in range(self.dim)]


@dataclass
class VGGishExtractorConfig(ERTKConfig):
    """Configuration for a VGGish extractor."""


class VGGishExtractor(
    FeatureExtractor, AudioClipProcessor, fname="vggish", config=VGGishExtractorConfig
):
    """A VGGish extractor."""

    config: VGGishExtractorConfig

    def __init__(self, config: VGGishExtractorConfig) -> None:
        super().__init__(config)

        from ertk.tensorflow import init_gpu_memory_growth

        init_gpu_memory_growth()

        import tensorflow_hub as hub

        self.model = hub.load(
            "https://www.kaggle.com/models/google/vggish/frameworks/TensorFlow2/variations/vggish/versions/1"  # noqa
        )

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if not is_mono_audio(x):
            raise ValueError("VGGish extractor requires mono audio input")
        embeddings = self.model(x)
        return np.mean(embeddings.numpy(), 0)

    @property
    def dim(self) -> int:
        return 128

    @property
    def is_sequence(self) -> bool:
        return False

    @property
    def feature_names(self) -> list[str]:
        return [f"vggish_{i}" for i in range(self.dim)]
