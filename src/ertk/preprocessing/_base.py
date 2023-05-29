import logging
import warnings
from abc import ABC, abstractmethod
from itertools import chain, tee
from typing import ClassVar, Dict, Iterable, List, Optional, Type, Union, cast

import librosa
import numpy as np
from importlib_metadata import EntryPoint, entry_points
from omegaconf import OmegaConf

from ertk.config import ERTKConfig
from ertk.utils import PathOrStr, batch_iterable


class InstanceProcessor(ABC):
    """An instance processor.

    Parameters
    ----------
    config: ERTKConfig
        The configuration for this processor.
    """

    config: ERTKConfig
    """The configuration for this processor."""
    logger: logging.Logger
    """The logger for this processor."""

    _config_type: ClassVar[Type[ERTKConfig]]
    _friendly_name: ClassVar[str]
    _registry: ClassVar[Dict[str, Type["InstanceProcessor"]]] = {}
    _plugins: ClassVar[Dict[str, EntryPoint]] = {
        x.name: x for x in entry_points().select(group="ertk.processors")
    }

    def __init__(self, config: ERTKConfig) -> None:
        self.config = config
        cls_name = f"{type(self).__module__}.{type(self).__name__}"
        self.logger = logging.getLogger(cls_name)

    def __init_subclass__(
        cls, fname: Optional[str] = None, config: Optional[Type[ERTKConfig]] = None
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

    def __repr__(self):
        return f"{type(self).__name__}({self.config})"

    def __str__(self):
        yaml = OmegaConf.to_yaml(self.config).strip()
        yaml = "\t" + yaml.replace("\n", "\n\t")
        return f"{type(self)._friendly_name}(\n{yaml}\n)"

    @classmethod
    def friendly_name(cls) -> str:
        """Get the friendly name for this processor."""
        return cls._friendly_name

    @classmethod
    def get_processor_class(cls, name: str) -> "Type[InstanceProcessor]":
        """Get the class for the named processor.

        Parameters
        ----------
        name: str
            The name of the processor.

        Returns
        -------
        processor: Type[InstanceProcessor]
            The processor class.
        """
        if name not in cls._registry:
            ep = cls._plugins.get(name)
            if ep is None:
                raise KeyError(f"Unknown processor: {name}")
            try:
                obj = ep.load()
            except ImportError as e:
                raise ImportError(f"Plugin {ep.name} could not be loaded.") from e
            if not issubclass(obj, InstanceProcessor):
                raise TypeError(
                    f"Plugin {ep.name} of type {obj} does not subclass "
                    "InstanceProcessor"
                )
            cls._registry[name] = obj
        return cls._registry[name]

    @classmethod
    def make_processor(cls, name: str, config: ERTKConfig) -> "InstanceProcessor":
        """Create an instance of the named processor.

        Parameters
        ----------
        name: str
            The name of the processor to create.
        config: ERTKConfig
            The configuration for the processor.

        Returns
        -------
        processor: InstanceProcessor
            The created processor.
        """
        return cls.get_processor_class(name)(config)

    @classmethod
    def get_config_type(cls) -> Type[ERTKConfig]:
        """Get the configuration type for this processor."""
        return cls._config_type

    @classmethod
    def valid_processors(cls) -> List[str]:
        """Get a list of all registered processor names."""
        return sorted(cls._registry.keys() | cls._plugins.keys())

    @classmethod
    def get_default_config(cls) -> ERTKConfig:
        """Get the default configuration for this processor."""
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
                self.process_batch(batch, **kwargs)
                for batch in batch_iterable(xs, batch_size)
            )
        elif batch_size < 0:
            yield from self.process_batch(xs, **kwargs)
        else:
            raise ValueError("Batch size cannot be 0.")

    def finish(self) -> None:
        """Perform any cleanup necesasry (e.g. closing files, unloading
        models, etc.)
        """

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        raise NotImplementedError()


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
    def dim(self) -> int:
        """The dimensionality of the extracted features."""
        return len(self.feature_names)

    @property
    @abstractmethod
    def is_sequence(self) -> bool:
        """Whether this FeatureExtractor yields sequence features."""
        raise NotImplementedError()
