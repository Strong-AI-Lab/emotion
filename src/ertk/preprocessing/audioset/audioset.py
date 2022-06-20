from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
from omegaconf import MISSING

from ertk.config import ERTKConfig
from ertk.preprocessing._base import AudioClipProcessor, FeatureExtractor
from ertk.preprocessing.spectrogram import spectrogram
from ertk.tensorflow.utils import init_gpu_memory_growth
from ertk.utils import frame_array, is_mono_audio


def _spectrogram(x: np.ndarray, sr: float) -> np.ndarray:
    return spectrogram(
        x,
        sr,
        kind="mel",
        pre_emphasis=0,
        window_size=0.025,
        window_shift=0.01,
        n_mels=64,
        fmin=125,
        fmax=7500,
    )


@dataclass
class YamnetExtractorConfig(ERTKConfig):
    model_dir: str = MISSING


class YamnetExtractor(
    FeatureExtractor, AudioClipProcessor, fname="yamnet", config=YamnetExtractorConfig
):
    config: YamnetExtractorConfig

    def __init__(self, config: YamnetExtractorConfig) -> None:
        super().__init__(config)

        import tensorflow as tf

        from ._yamnet.params import Params
        from ._yamnet.yamnet import yamnet

        init_gpu_memory_growth()

        input_layer = tf.keras.Input((96, 64))
        predictions, embeddings = yamnet(input_layer, Params())
        model = tf.keras.Model(inputs=input_layer, outputs=[predictions, embeddings])
        model.load_weights(str(Path(config.model_dir, "yamnet.h5")))
        model(tf.zeros((1, 96, 64)))
        self.model = model

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if is_mono_audio(x):
            x = _spectrogram(x, kwargs.pop("sr"))
        frames = frame_array(x, 96, 48, pad=True, axis=0)
        _, embeddings = self.model(frames)
        return np.mean(embeddings, 0)

    def process_batch(
        self, batch: Union[Iterable[np.ndarray], np.ndarray], **kwargs
    ) -> List[np.ndarray]:
        batch = list(batch)
        if all(is_mono_audio(x) for x in batch):
            sr = kwargs.pop("sr")
            batch = [_spectrogram(x, sr) for x in batch]
        frames = [frame_array(spec, 96, 48, pad=True, axis=0) for spec in batch]
        slices = np.cumsum([len(x) for x in frames])[:-1]
        frames = np.concatenate(frames)
        frames *= np.log(10.0) / 20  # Rescale dB power to ln(abs(X))

        _, embeddings = self.model(frames)
        return [np.mean(x, 0) for x in np.split(embeddings, slices)]

    @property
    def dim(self) -> int:
        return 1024

    @property
    def is_sequence(self) -> bool:
        return False

    @property
    def feature_names(self) -> List[str]:
        return [f"yamnet_{i}" for i in range(self.dim)]


@dataclass
class VGGishExtractorConfig(ERTKConfig):
    model_dir: str = MISSING


class VGGishExtractor(
    FeatureExtractor, AudioClipProcessor, fname="vggish", config=VGGishExtractorConfig
):
    config: VGGishExtractorConfig

    def __init__(self, config: VGGishExtractorConfig) -> None:
        super().__init__(config)

        import tensorflow.compat.v1 as tf

        from ._vggish.vggish_postprocess import Postprocessor
        from ._vggish.vggish_slim import define_vggish_slim, load_vggish_slim_checkpoint

        tf.disable_eager_execution()
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        pca_params = str(Path(config.model_dir, "vggish_pca_params.npz"))
        self.processor = Postprocessor(pca_params)

        self.sess = tf.Session()

        define_vggish_slim()
        ckpt = str(Path(config.model_dir, "vggish_model.ckpt"))
        load_vggish_slim_checkpoint(self.sess, ckpt)

        self.features_tensor = self.sess.graph.get_tensor_by_name(
            "vggish/input_features:0"
        )
        self.embedding_tensor = self.sess.graph.get_tensor_by_name("vggish/embedding:0")

    def __del__(self):
        self.sess.close()

    def _process_frames(self, frames: np.ndarray) -> np.ndarray:
        [embeddings] = self.sess.run(
            [self.embedding_tensor], feed_dict={self.features_tensor: frames}
        )
        return self.processor.postprocess(embeddings)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if is_mono_audio(x):
            x = _spectrogram(x, kwargs.pop("sr"))
        frames = frame_array(x, 96, 96, pad=True, axis=0)
        return np.mean(self._process_frames(frames), 0)

    def process_batch(
        self, batch: Union[Iterable[np.ndarray], np.ndarray], **kwargs
    ) -> List[np.ndarray]:
        batch = list(batch)
        if all(is_mono_audio(x) for x in batch):
            sr = kwargs.pop("sr")
            batch = [_spectrogram(x, sr) for x in batch]
        frames = [frame_array(spec, 96, 96, pad=True, axis=0) for spec in batch]
        slices = np.cumsum([len(x) for x in frames])[:-1]
        frames = np.concatenate(frames)
        frames *= np.log(10.0) / 20  # Rescale dB power to ln(abs(X))
        return [np.mean(x, 0) for x in np.split(self._process_frames(frames), slices)]

    @property
    def dim(self) -> int:
        return 128

    @property
    def is_sequence(self) -> bool:
        return False

    @property
    def feature_names(self) -> List[str]:
        return [f"vggish_{i}" for i in range(self.dim)]
