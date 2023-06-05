"""Feature extractor using Keras applications."""

from dataclasses import dataclass
from types import ModuleType
from typing import Iterable, List, Optional, Union

import numpy as np
from omegaconf import MISSING
from scipy.ndimage import zoom

from ertk.config import ERTKConfig
from ertk.preprocessing.spectrogram import spectrogram
from ertk.utils import is_mono_audio

from ._base import AudioClipProcessor, FeatureExtractor

__all__ = ["KerasAppsExtractorConfig", "KerasAppsExtractor"]


@dataclass
class _ModelArgs:
    module: ModuleType
    funcname: str
    size: int


MODEL_MAP = {}


def _init_models():
    from keras.applications import (
        densenet,
        efficientnet,
        inception_resnet_v2,
        inception_v3,
        mobilenet,
        mobilenet_v2,
        mobilenet_v3,
        nasnet,
        resnet,
        vgg16,
        vgg19,
        xception,
    )

    global MODEL_MAP
    MODEL_MAP = {
        "densenet121": _ModelArgs(densenet, "DenseNet121", 224),
        "densenet169": _ModelArgs(densenet, "DenseNet169", 224),
        "densenet201": _ModelArgs(densenet, "DenseNet201", 224),
        "efficientnet_b0": _ModelArgs(efficientnet, "EfficientNetB0", 224),
        "efficientnet_b1": _ModelArgs(efficientnet, "EfficientNetB1", 240),
        "efficientnet_b2": _ModelArgs(efficientnet, "EfficientNetB2", 260),
        "efficientnet_b3": _ModelArgs(efficientnet, "EfficientNetB3", 300),
        "efficientnet_b4": _ModelArgs(efficientnet, "EfficientNetB4", 380),
        "efficientnet_b5": _ModelArgs(efficientnet, "EfficientNetB5", 456),
        "efficientnet_b6": _ModelArgs(efficientnet, "EfficientNetB6", 528),
        "efficientnet_b7": _ModelArgs(efficientnet, "EfficientNetB7", 600),
        "inception_resnet_v2": _ModelArgs(
            inception_resnet_v2, "InceptionResNetV2", 299
        ),
        "inception_v3": _ModelArgs(inception_v3, "InceptionV3", 299),
        "mobilenet": _ModelArgs(mobilenet, "MobileNet", 224),
        "mobilenet_v2": _ModelArgs(mobilenet_v2, "MobileNetV2", 224),
        "mobilenet_v3_large": _ModelArgs(mobilenet_v3, "MobileNetV3Large", 224),
        "mobilenet_v3_small": _ModelArgs(mobilenet_v3, "MobileNetV3Small", 224),
        "nasnet_large": _ModelArgs(nasnet, "NASNetLarge", 331),
        "nasnet_mobile": _ModelArgs(nasnet, "NASNetMobile", 224),
        "resnet50": _ModelArgs(resnet, "ResNet50", 224),
        "resnet50_v2": _ModelArgs(resnet, "ResNet50V2", 224),
        "resnet101": _ModelArgs(resnet, "ResNet101", 224),
        "resnet101_v2": _ModelArgs(resnet, "ResNet101V2", 224),
        "resnet152": _ModelArgs(resnet, "ResNet152", 224),
        "resnet152_v2": _ModelArgs(resnet, "ResNet152V2", 224),
        "vgg16": _ModelArgs(vgg16, "VGG16", 224),
        "vgg19": _ModelArgs(vgg19, "VGG19", 224),
        "xception": _ModelArgs(xception, "Xception", 299),
    }


@dataclass
class KerasAppsExtractorConfig(ERTKConfig):
    """Configuration for KerasAppsExtractor."""

    model: str = MISSING
    """Name of the model to use. See https://keras.io/api/applications/
    for the list of available models.
    """
    size: Optional[int] = None
    """Size of the input image. If not specified, the default size of the
    model will be used.
    """
    n_mels: int = 128
    """Number of mel bands for spectrogram extraction."""
    spec_type: str = "mel"
    """Type of spectrogram to extract. Can be either 'mel' or 'linear'."""
    fmax: int = 8000
    """Maximum frequency for spectrogram extraction."""


class KerasAppsExtractor(
    FeatureExtractor,
    AudioClipProcessor,
    fname="keras_apps",
    config=KerasAppsExtractorConfig,
):
    """Feature extractor using Keras applications."""

    config: KerasAppsExtractorConfig

    def __init__(self, config: KerasAppsExtractorConfig) -> None:
        super().__init__(config)

        from ertk.tensorflow.utils import init_gpu_memory_growth

        init_gpu_memory_growth()
        _init_models()

        model_args = MODEL_MAP[config.model]
        size = config.size or model_args.size
        self.size = size
        args = {
            "include_top": False,
            "weights": "imagenet",
            "pooling": "avg",
            "input_shape": (size, size, 3),
        }
        self.model = getattr(model_args.module, model_args.funcname)(**args)
        self._preprocess_input = getattr(model_args.module, "preprocess_input")
        _test = self._spectrogram(np.zeros(16000, dtype=np.float32), 16000)
        _test = self._to_img(_test)
        _test = self._preprocess_input(_test)
        _test = self.model(_test[None, :, :, :]).numpy()
        self._dim = _test.shape[-1]

    def _spectrogram(self, x: np.ndarray, sr: float) -> np.ndarray:
        return spectrogram(
            x,
            sr=sr,
            kind=self.config.spec_type,
            pre_emphasis=0,
            window_size=0.025,
            window_shift=0.01,
            n_mels=self.config.n_mels,
            fmin=0,
            fmax=None,
            power=2,
            to_log="db",
        )

    def _to_img(self, spec: np.ndarray) -> np.ndarray:
        if spec.ndim > 2:
            spec = spec.squeeze(0)
        scale = self.size / np.array(spec.shape)
        spec = zoom(spec, scale)
        spec = np.interp(spec, (spec.min(), spec.max()), (0, 255))
        return np.tile(spec[:, :, None], (1, 1, 3))

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if is_mono_audio(x):
            x = self._spectrogram(x.squeeze(), kwargs.pop("sr"))
        x = self._to_img(x)
        x = self._preprocess_input(x)
        return self.model(x[None, :, :, :]).numpy()

    def process_batch(
        self, batch: Union[Iterable[np.ndarray], np.ndarray], **kwargs
    ) -> List[np.ndarray]:
        batch = list(batch)
        if any(is_mono_audio(x) for x in batch):
            if any(not is_mono_audio(x) for x in batch):
                raise RuntimeError("Inconsistent input representation")
            sr = kwargs.pop("sr")
            batch = [self._spectrogram(x.squeeze(), sr) for x in batch]
        batch = [self._to_img(x) for x in batch]
        batch = np.stack(batch)
        batch = self._preprocess_input(batch)
        return list(self.model(batch).numpy())

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_sequence(self) -> bool:
        return False

    @property
    def feature_names(self) -> List[str]:
        return [f"{self.config.model}_{i}" for i in range(self.dim)]
