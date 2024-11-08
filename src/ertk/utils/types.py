"""type definitions."""

from collections.abc import Callable
from os import PathLike
from typing import Union

import numpy as np

__all__ = ["PathOrStr", "ScoreFunction"]


PathOrStr = Union[PathLike, str]
ScoreFunction = Callable[[np.ndarray, np.ndarray], float]
