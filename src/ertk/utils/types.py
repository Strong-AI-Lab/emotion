"""type definitions."""

from collections.abc import Callable
from os import PathLike

import numpy as np

__all__ = ["PathOrStr", "ScoreFunction"]


PathOrStr = PathLike | str
ScoreFunction = Callable[[np.ndarray, np.ndarray], float]
