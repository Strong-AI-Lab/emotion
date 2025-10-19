"""TensorFlow training configuration.

.. autosummary::
    :toctree:

    TFLoggingConfig
    TFTrainConfig
"""

from dataclasses import dataclass

from ertk.config import ERTKConfig

__all__ = ["TFLoggingConfig", "TFTrainConfig"]


@dataclass
class TFLoggingConfig(ERTKConfig):
    log_dir: str | None = None
    name: str = ""
    version: str = ""
    tensorboard: bool = True
    csv: bool = True


@dataclass
class TFTrainConfig(ERTKConfig):
    batch_size: int = 32
    logging: TFLoggingConfig = TFLoggingConfig()
    n_gpus: int = 1
    epochs: int = 50
    data_fn: str | None = None
