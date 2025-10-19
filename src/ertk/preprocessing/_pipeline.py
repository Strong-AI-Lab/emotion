from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from omegaconf import MISSING

from ertk.config import ERTKConfig
from ertk.utils import PathOrStr

from ._base import InstanceProcessor


@dataclass
class ProcessorConfig:
    """Configuration for a processor."""

    type: str = MISSING
    """The type of processor to use (friendly name)."""
    batch_size: int = 1
    """The batch size to use for processing."""
    config: dict[str, Any] = field(default_factory=lambda: {})
    """The processor configuration."""


class ChainingType(Enum):
    """The type of chaining to use for a processing pipeline."""

    IN_MEMORY = "in_memory"
    """Results from each processor are stored in memory."""
    GENERATOR = "generator"
    """Results from each processor are yielded as they are produced."""


@dataclass
class ProcessingPipelineConfig(ERTKConfig):
    """Configuration for a processing pipeline."""

    input: str = MISSING
    """The input data to process."""
    output: str = MISSING
    """Where to write the output data."""
    chaining: ChainingType = ChainingType.IN_MEMORY
    """The type of chaining to use for the pipeline"""
    pipeline: list[str] = MISSING
    """The processors to use in the pipeline, in order"""
    processors: dict[str, ProcessorConfig] = MISSING
    """The processors to use in the pipeline, and their configurations"""


class ProcessingPipeline(InstanceProcessor):
    """A processing pipeline.

    Parameters
    ----------
    config: ProcessingPipelineConfig
        The configuration for the pipeline.
    """

    pipeline: list[InstanceProcessor]
    """The processors in the pipeline."""
    chaining: ChainingType = ChainingType.IN_MEMORY
    """The type of chaining to use for the pipeline"""
    output: str
    """Where to write the output data."""
    config: ProcessingPipelineConfig
    """The pipeline configuration."""

    def __init__(self, config: ProcessingPipelineConfig) -> None:
        from ertk.dataset.features import read_features_iterable

        data = read_features_iterable(config.input)
        self.data = data
        self.config = config
        self.chaining = config.chaining
        self.output = config.output
        self.pipeline = []
        for name in config.pipeline:
            # This implicitly checks that the processor exists, and that
            # the config is valid
            proc_conf = (
                InstanceProcessor.get_processor_class(name)
                .get_config_type()
                .from_config(config.processors[name].config)
            )
            processor = InstanceProcessor.make_processor(name, proc_conf)
            self.pipeline.append(processor)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        for processor in self.pipeline:
            x = processor.process_instance(x, **kwargs)
        return x

    def process_batch(
        self, batch: Iterable[np.ndarray] | np.ndarray, **kwargs
    ) -> list[np.ndarray]:
        for processor in self.pipeline:
            batch = processor.process_batch(batch, **kwargs)
        return list(batch)

    def process_all(
        self, xs: Iterable[np.ndarray] | np.ndarray, batch_size: int, **kwargs
    ) -> Iterable[np.ndarray]:
        for processor in self.pipeline:
            xs = processor.process_all(xs, batch_size, **kwargs)
            if self.chaining == ChainingType.IN_MEMORY:
                xs = list(xs)
        return xs

    def finish(self) -> None:
        for processor in self.pipeline:
            processor.finish()

    @property
    def feature_names(self) -> list[str]:
        return self.pipeline[-1].feature_names


def load_pipeline(file: PathOrStr) -> ProcessingPipeline:
    """Load a processing pipeline from a YAML file.

    Parameters
    ----------
    file: pathlike or str
        The path to the YAML file.

    Returns
    -------
    pipeline: ProcessingPipeline
        The loaded processing pipeline.
    """

    config = ProcessingPipelineConfig.from_file(file)
    return ProcessingPipeline(config)
