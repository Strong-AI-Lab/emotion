"""Training configuration for PyTorch models.

.. autosummary::
    :toctree:

    PyTorchTrainConfig
    PyTorchLoggingConfig
    PyTorchDataConfig
    PyTorchDataAugConfig
"""

from dataclasses import dataclass, field
from typing import Any

import omegaconf

from ertk.config import ERTKConfig
from ertk.pytorch.models import PyTorchModelConfig

__all__ = [
    "PyTorchTrainConfig",
    "PyTorchLoggingConfig",
    "PyTorchDataConfig",
    "PyTorchDataAugConfig",
]


@dataclass
class PyTorchLoggingConfig(ERTKConfig):
    log_dir: str | None = ""
    tensorboard: bool = True
    csv: bool = True


@dataclass
class PyTorchDataAugConfig(ERTKConfig):
    spec: str = omegaconf.MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PyTorchDataConfig(ERTKConfig):
    train_data_processing: dict[str, PyTorchDataAugConfig] = field(default_factory=dict)
    valid_data_processing: dict[str, PyTorchDataAugConfig] = field(default_factory=dict)
    test_data_processing: dict[str, PyTorchDataAugConfig] = field(default_factory=dict)


@dataclass
class PyTorchTrainConfig(ERTKConfig):
    batch_size: int = 32
    logging: PyTorchLoggingConfig = PyTorchLoggingConfig()
    n_gpus: int = 1
    dist_strategy: str = "dp"
    epochs: int = 50
    wrapper_model_config: PyTorchModelConfig = PyTorchModelConfig()
    enable_checkpointing: bool = False
    resume_checkpoint: str = ""
    save_every_n_epochs: int = 1
    data_config: PyTorchDataConfig = PyTorchDataConfig()
