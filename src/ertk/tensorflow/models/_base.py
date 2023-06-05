from dataclasses import dataclass, field
from typing import Any, Dict

from ertk.config import ERTKConfig

__all__ = ["TFModelConfig"]


@dataclass
class TFModelConfig(ERTKConfig):
    """Configuration for a TensorFlow model."""

    optimiser: str = "adam"
    """The optimiser to use."""
    opt_params: Dict[str, Any] = field(default_factory=dict)
    """The parameters to pass to the optimiser."""
    learning_rate: float = 1e-3
    """The learning rate to use."""
    n_features: int = -1
    """The number of features in the input data."""
    n_classes: int = -1
    """The number of classes in the output data."""
    loss: str = "sparse_categorical_crossentropy"
    """The loss function to use."""
