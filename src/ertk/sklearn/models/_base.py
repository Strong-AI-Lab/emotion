import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Type, cast

import yaml
from omegaconf import MISSING, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler

from ertk.config import ERTKConfig

CLASSIFIER_MAP = {
    "rf": RandomForestClassifier,
    "knn": KNeighborsClassifier,
    "lr": LogisticRegression,
    "mlp": MLPClassifier,
}

TRANSFORM_MAP = {
    "std": StandardScaler,
    "minmax": MinMaxScaler,
    "norm": Normalizer,
    "maxabs": MaxAbsScaler,
}


@dataclass
class SkModelConfig(ERTKConfig):
    kwargs: Dict[str, Any] = field(default_factory=dict)
    transform: Optional[str] = "std"
    transform_kwargs: Dict[str, Any] = field(default_factory=dict)
    param_grid: Dict[str, Any] = field(default_factory=dict)
    param_grid_path: Optional[str] = None


class ERTKSkModel(ABC):
    config: SkModelConfig

    _config_type: ClassVar[Type[SkModelConfig]]
    _friendly_name: ClassVar[str]
    _registry: ClassVar[Dict[str, Type["ERTKSkModel"]]] = {}

    def __init__(self, config: SkModelConfig) -> None:
        full_config = OmegaConf.merge(type(self).get_default_config(), config)
        # Inplace merge so that subclass __init__() also gets the full config
        config.merge_with(full_config)  # type: ignore

        self.config = config

    def __init_subclass__(
        cls, fname: str = None, config: Type[SkModelConfig] = None
    ) -> None:
        cls._registry = {}
        if fname and config:
            cls._friendly_name = fname
            cls._config_type = config
            for t in cls.mro()[1:-1]:
                t = cast(Type[ERTKSkModel], t)  # For MyPy
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

    @classmethod
    def get_model_class(cls, name: str) -> Type["ERTKSkModel"]:
        try:
            return cls._registry[name]
        except KeyError as e:
            raise ValueError(f"No model named {name}") from e

    @classmethod
    def make_model(cls, name: str, config: SkModelConfig) -> "ERTKSkModel":
        return cls.get_model_class(name)(config)

    @classmethod
    def get_config_type(cls) -> Type[SkModelConfig]:
        return cls._config_type

    @classmethod
    def valid_models(cls) -> List[str]:
        return list(cls._registry)

    @classmethod
    def get_default_config(cls) -> SkModelConfig:
        return OmegaConf.structured(cls._config_type)


class SkWrapperConfig(SkModelConfig):
    kind: str = MISSING


class SkWrapperModel(ERTKSkModel, fname="wrapper", config=SkWrapperConfig):
    config: SkWrapperConfig

    def __init__(self, config: SkWrapperConfig) -> None:
        super().__init__(config)

        self.kind = config.kind
        clf = CLASSIFIER_MAP[self.kind](**config.kwargs)

        transform = None
        if config.transform:
            transform = TRANSFORM_MAP[config.transform](**config.transform_kwargs)
        clf = Pipeline([("transform", transform), ("clf", clf)])

        param_grid = {}
        if config.param_grid_path:
            with open(config.param_grid_path) as fid:
                param_grid = yaml.safe_load(fid)
        elif config.param_grid:
            param_grid = config.param_grid
        if len(ParameterGrid(param_grid)) > 1:
            clf = GridSearchCV(clf, param_grid)

        self.clf = clf


def gen_wrapper_for_clf(kind):
    class _(ERTKSkModel, fname=kind, config=SkModelConfig):
        def __init__(self, config: SkModelConfig) -> None:
            super().__init__(config)
            self.kind = kind
            clf = CLASSIFIER_MAP[kind](**config.kwargs)

            transform = None
            if config.transform:
                transform = TRANSFORM_MAP[config.transform](**config.transform_kwargs)
            clf = Pipeline([("transform", transform), ("clf", clf)])

            param_grid = {}
            if config.param_grid_path:
                with open(config.param_grid_path) as fid:
                    param_grid = yaml.safe_load(fid)
            elif config.param_grid:
                param_grid = config.param_grid
            if len(ParameterGrid(param_grid)) > 1:
                clf = GridSearchCV(clf, param_grid)

            self.clf = clf


for key in CLASSIFIER_MAP:
    gen_wrapper_for_clf(key)
