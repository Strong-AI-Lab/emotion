"""
Configuration utilities
=======================

This module provides utilities for working with configuration
dataclasses and YAML files.


Base classes
------------
.. autosummary::
    :toctree: generated/

    ERTKConfig


Functions
---------
.. autosummary::
    :toctree: generated/

    get_arg_mapping
"""

from abc import ABC
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from omegaconf import DictConfig, OmegaConf

from ertk.utils import PathOrStr

__all__ = ["ERTKConfig", "get_arg_mapping"]


T = TypeVar("T", bound="ERTKConfig")


def _resolve_files(key):  # pragma: no cover
    return key


def _resolve_files_load(key):  # pragma: no cover
    return OmegaConf.load(key)


OmegaConf.register_new_resolver("file", _resolve_files)
OmegaConf.register_new_resolver("cwdpath", _resolve_files_load)


@dataclass
class ERTKConfig(ABC):
    """Base class for ERTK configuration dataclasses."""

    def to_dictconfig(self) -> DictConfig:
        """Convert config to DictConfig.

        Returns
        -------
        omegaconf.DictConfig
            The structured config which behaves like the corresponding
            dataclass.
        """
        return DictConfig(self)

    @classmethod
    def from_config(cls: type[T], config: Any) -> T:
        """Create config object from any compatible config.

        Parameters
        ----------
        config: omegaconf.DictConfig or dict
            The object to create the config from.

        Returns
        -------
        ERKConfig
            The resulting config.
        """
        return cast(T, OmegaConf.merge(OmegaConf.structured(cls), config))

    @classmethod
    def default(cls: type[T]) -> T:
        """Create default config.

        Returns
        -------
        ERKConfig
            The default config.
        """
        return cast(T, OmegaConf.structured(cls))

    @classmethod
    def from_file(cls: type[T], path: PathOrStr) -> T:
        """Create config from YAML file and optionlly override some
        values.

        Parameters
        ----------
        path: os.Pathlike or str
            The path to YAML file containing config.
        override: collection of str, optional
            Argument overrides in the form of key=value pairs.

        Returns
        -------
        ERKConfig
            The resulting config.
        """
        return cls.from_config(OmegaConf.load(Path(path)))

    def to_string(self) -> str:
        """Generate YAML string representation of config.

        Returns
        -------
        str
            YAML string representation of config.
        """
        return OmegaConf.to_yaml(self)

    def to_file(self, path: PathOrStr) -> None:
        """Write config to YAML file.

        Parameters
        ----------
        path: os.Pathlike or str
            The path to YAML file to write config to.
        """
        with open(path, "w") as fid:
            fid.write(OmegaConf.to_yaml(self))

    def merge_with_config(self: T, config: Any) -> T:
        """Merge other config into this config.

        Parameters
        ----------
        config: omegaconf.DictConfig or dict
            The object to create the config from.

        Returns
        -------
        ERKConfig
            The resulting config.
        """
        return cast(T, OmegaConf.merge(self, config))

    def merge_with_args(self: T, args: Collection[str]) -> T:
        """Merge config with command-line arguments.

        Parameters
        ----------
        args: collection of str
            Argument overrides in the form of key=value pairs.

        Returns
        -------
        ERKConfig
            The resulting config.
        """
        return cast(T, OmegaConf.merge(self, OmegaConf.from_dotlist(list(args))))


def get_arg_mapping(s: Path | str) -> dict[str, str]:
    """Given a mapping on the command-line, returns a dict representing
    that mapping. Mapping can be a string or a more complex YAML file.

    The string form of the mapping is:
        key:value[,key:value]+

    Parameters
    ----------
    s: PathLike or str
        String representing the mapping or path to YAML containing
        mapping. If a string, it cannot contain spaces or shell symbols
        (unless escaped).

    Returns
    -------
    dict
        A dictionary mapping keys to values from the string.
    """
    if isinstance(s, Path) or Path(s).exists():
        with open(s) as fid:
            return yaml.safe_load(fid) or {}
    mapping: dict[str, list[str]] = defaultdict(list)
    for cls in s.split(","):
        key, val = cls.split(":")
        mapping[key].append(val)
    return {k: v[0] if len(v) == 1 else v for k, v in mapping.items()}
