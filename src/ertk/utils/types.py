from os import PathLike
from typing import Callable, Union

import numpy as np


PathOrStr = Union[PathLike, str]
ScoreFunction = Callable[[np.ndarray, np.ndarray], float]
