import os
import subprocess
import tempfile
from dataclasses import dataclass
from importlib.resources import path as res_path
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
from omegaconf import MISSING

from ertk.config import ERTKConfig
from ertk.dataset import write_features
from ertk.preprocessing._base import FeatureExtractor


@dataclass
class OpenXBOWExtractorConfig(ERTKConfig):
    xbowargs: List[str] = MISSING


class OpenXBOWExtractor(
    FeatureExtractor, fname="openxbow", config=OpenXBOWExtractorConfig
):
    config: OpenXBOWExtractorConfig
    _dim = None

    def __init__(self, config: OpenXBOWExtractorConfig) -> None:
        super().__init__(config)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def process_batch(
        self,
        batch: Union[Iterable[np.ndarray], np.ndarray],
        **kwargs,
    ) -> List[np.ndarray]:

        _, tmpin = tempfile.mkstemp(prefix="openxbow_", suffix=".csv")
        _, tmpout = tempfile.mkstemp(prefix="openxbow_", suffix=".csv")

        xs = list(batch)
        names = [str(i) for i in range(len(xs))]
        write_features(tmpin, xs, names=names)

        with res_path("ertk.preprocessing.openxbow", "openXBOW.jar") as jar:
            xbow_args = [
                "java",
                "-jar",
                str(jar),
                "-i",
                tmpin,
                "-o",
                tmpout,
                "-attributes",
                f"n1[{xs[0].shape[1]}]",
                "-csvSep",
                ",",
                "-writeName",
                "-noLabels",
                *self.config.xbowargs,
            ]

        if subprocess.call(xbow_args) != 0:
            raise RuntimeError("openXBOW returned an error")
        os.remove(tmpin)
        data: pd.DataFrame = pd.read_csv(
            tmpout, header=None, quotechar="'", dtype={0: str}
        )
        os.remove(tmpout)
        self._dim = data.shape[1] - 1

        return list(data.iloc[:, 1:].to_numpy())

    @property
    def dim(self) -> int:
        for i, arg in enumerate(self.config.xbowargs):
            if arg.startswith("-size"):
                try:
                    _, size = arg.split("=", maxsplit=1)
                except ValueError:
                    size = self.config.xbowargs[i + 1]
                return int(size)
        if self._dim:
            return self._dim
        raise ValueError("Cannot determine dim.")

    @property
    def is_sequence(self) -> bool:
        return False

    @property
    def feature_names(self) -> List[str]:
        return [f"bow_{i}" for i in range(self.dim)]
