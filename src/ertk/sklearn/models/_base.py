import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, cast

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
    """Configuration for a scikit-learn model."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass to the model constructor."""
    transform: Optional[str] = "std"
    """The name of the transform to use."""
    transform_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass to the transform constructor."""
    param_grid: dict[str, Any] = field(default_factory=dict)
    """The parameter grid to use for grid search."""
    param_grid_path: Optional[str] = None
    """The path to a YAML file containing the parameter grid to use for
    grid search.
    """


class ERTKSkModel(ABC):
    """Base class for scikit-learn models.

    Parameters
    ----------
    config: SkModelConfig
        The model configuration.
    """

    config: SkModelConfig
    """The model configuration."""

    _config_type: ClassVar[type[SkModelConfig]]
    _friendly_name: ClassVar[str]
    _registry: ClassVar[dict[str, type["ERTKSkModel"]]] = {}

    def __init__(self, config: SkModelConfig) -> None:
        full_config = OmegaConf.merge(type(self).get_default_config(), config)
        # Inplace merge so that subclass __init__() also gets the full config
        config.merge_with(full_config)  # type: ignore

        self.config = config

    def __init_subclass__(
        cls, fname: str = None, config: type[SkModelConfig] = None
    ) -> None:
        cls._registry = {}
        if fname and config:
            cls._friendly_name = fname
            cls._config_type = config
            for t in cls.mro()[1:-1]:
                t = cast(type[ERTKSkModel], t)  # For MyPy
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
    def get_model_class(cls, name: str) -> type["ERTKSkModel"]:
        try:
            return cls._registry[name]
        except KeyError as e:
            raise ValueError(f"No model named {name}") from e

    @classmethod
    def make_model(cls, name: str, config: SkModelConfig) -> "ERTKSkModel":
        return cls.get_model_class(name)(config)

    @classmethod
    def get_config_type(cls) -> type[SkModelConfig]:
        return cls._config_type

    @classmethod
    def valid_models(cls) -> list[str]:
        return list(cls._registry)

    @classmethod
    def get_default_config(cls) -> SkModelConfig:
        return OmegaConf.structured(cls._config_type)


class SkWrapperConfig(SkModelConfig):
    """Configuration for a scikit-learn model wrapper."""

    kind: str = MISSING
    """The name of the model to wrap."""


class SkWrapperModel(ERTKSkModel, fname="wrapper", config=SkWrapperConfig):
    """A wrapper for scikit-learn models."""

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


def gen_wrapper_for_clf(kind: str) -> None:
    """Generate a wrapper for a scikit-learn classifier.

    Parameters
    ----------
    kind: str
        The name of the classifier.
    """

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
