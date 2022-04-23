import logging
import warnings
from abc import ABC, abstractmethod
from itertools import chain, tee
from typing import ClassVar, Dict, Iterable, List, Optional, Type, Union, cast

import librosa
import numpy as np
from omegaconf import OmegaConf

from ertk.config import ERTKConfig
from ertk.utils import PathOrStr, batch_iterable


class InstanceProcessor(ABC):
    """An instance processor.

    Parameters
    ----------
    config: ERTKConfig
        The configuration for this AudioClipProcessor.
    """

    config: ERTKConfig
    """The configuration for this AudioClipProcessor."""
    logger: logging.Logger

    _config_type: ClassVar[Type[ERTKConfig]]
    _friendly_name: ClassVar[str]
    _registry: ClassVar[Dict[str, Type["InstanceProcessor"]]] = {}

    def __init__(self, config: ERTKConfig) -> None:
        self.config = config
        cls_name = f"{InstanceProcessor.__module__}.{InstanceProcessor.__name__}"
        self.logger = logging.getLogger(cls_name)

    def __init_subclass__(
        cls, fname: str = None, config: Type[ERTKConfig] = None
    ) -> None:
        cls._registry = {}
        if fname and config:
            cls._friendly_name = fname
            cls._config_type = config
            for t in cls.mro()[1:-1]:
                t = cast(Type[InstanceProcessor], t)  # For MyPy
                if not hasattr(t, "_registry"):
                    continue
                if fname in t._registry:
                    prev_cls = t._registry[fname]
                    msg = f"Name {fname} already registered with class {prev_cls}."
                    if prev_cls is cls:
                        warnings.warn(msg)
                    else:
                        raise KeyError(msg)
                t._registry[fname] = cls

    @classmethod
    def get_processor_class(cls, name: str) -> Type["InstanceProcessor"]:
        return cls._registry[name]

    @classmethod
    def make_processor(cls, name: str, config: ERTKConfig) -> "InstanceProcessor":
        return cls.get_processor_class(name)(config)

    @classmethod
    def get_config_type(cls) -> Type[ERTKConfig]:
        return cls._config_type

    @classmethod
    def valid_preprocessors(cls) -> List[str]:
        return list(cls._registry)

    @classmethod
    def get_default_config(cls):
        return OmegaConf.structured(cls._config_type)

    @abstractmethod
    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Process a single audio clip.

        Parameters
        ----------
        x: np.ndarray
            The audio data to process.

        Returns
        -------
        result: np.ndarray
            The processed instance.
        """
        raise NotImplementedError()

    def process_batch(
        self, batch: Union[Iterable[np.ndarray], np.ndarray], **kwargs
    ) -> List[np.ndarray]:
        """Process a batch of instances. By default this simply calls
        `process_instance()` on each instance in the batch.

        Parameters
        ----------
        batch: np.ndarray or iterable of arrays
            The batch of instances to process.

        Returns
        -------
        processed: list of np.ndarray
            A list of processed instances.
        """
        return [self.process_instance(x, **kwargs) for x in batch]

    def process_all(
        self, xs: Union[Iterable[np.ndarray], np.ndarray], batch_size: int, **kwargs
    ) -> Iterable[np.ndarray]:
        """Process all instances in batches.

        Parameters
        ----------
        xs: iterable of np.ndarray
            The data from instances to process.
        batch_size: int
            Batch size.

        Returns
        -------
        processed: iterable of np.ndarray
            A generator that yields each processed instance in order.
        """
        if batch_size == 1:
            for x in xs:
                yield self.process_instance(x, **kwargs)
        elif batch_size > 1:
            yield from chain.from_iterable(
                self.process_batch(filter(lambda x: x is not None, b), **kwargs)
                for b in batch_iterable(xs, batch_size)
            )
        elif batch_size < 0:
            yield from self.process_batch(xs, **kwargs)
        else:
            raise ValueError("Batch size cannot be 0.")


class AudioClipProcessor(InstanceProcessor):
    """Processes raw audio data."""

    def process_file(self, path: PathOrStr, sr: Optional[float] = None) -> np.ndarray:
        """Process individual audio file.

        Parameters
        ----------
        path: pathlike or str
            The path to an audio file.
        sr: float, optional
            Target sample rate. If not given, then each audio file is
            loaded with its native sample rate. Otherwise audio data is
            resampled to this sample rate.

        Returns
        -------
        processed: np.ndarray
            Processed instance.
        """
        audio, _sr = librosa.load(path, sr=sr, mono=True)
        return self.process_instance(audio, sr=_sr)

    def process_files(
        self, paths: Iterable[PathOrStr], batch_size: int, sr: Optional[float] = None
    ) -> Iterable[np.ndarray]:
        """Process a set of files.

        Parameters
        ----------
        paths: iterables of path or iterable of str
            Paths to audio files.
        batch_size: int
            Batch size. Note that if `batch_size != 1` then either `sr`
            must be given, or all files must have the same native sample
            rate.
        sr: float, optional
            Target sample rate. If not given, then each audio file is
            loaded with its native sample rate. Otherwise audio data is
            resampled to this sample rate.

        Returns
        -------
        processed: iterable of np.ndarray
            Iterable of processed instances.
        """
        if batch_size == 1:
            # Special case required for variable sample rate
            for path in paths:
                yield self.process_file(path, sr=sr)
        else:
            paths, tmp = tee(paths)
            _sr = librosa.load(next(tmp), sr=sr, duration=0)[1]
            audios = (librosa.load(path, sr=_sr, mono=True)[0] for path in paths)
            yield from self.process_all(audios, batch_size=batch_size, sr=_sr)


class FeatureExtractor(InstanceProcessor):
    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimensionality of the extracted features."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_sequence(self) -> bool:
        """Whether this FeatureExtractor yields sequence features."""
        raise NotImplementedError()
