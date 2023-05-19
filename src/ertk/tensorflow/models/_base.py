from dataclasses import dataclass, field
from typing import Any, Dict

from ertk.config import ERTKConfig


@dataclass
class TFModelConfig(ERTKConfig):
    optimiser: str = "adam"
    opt_params: Dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 1e-3
    n_features: int = -1
    n_classes: int = -1
    loss: str = "sparse_categorical_crossentropy"
