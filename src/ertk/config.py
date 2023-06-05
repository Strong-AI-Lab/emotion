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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

import yaml
from omegaconf import DictConfig, OmegaConf

from ertk.utils import PathOrStr

__all__ = ["ERTKConfig", "get_arg_mapping"]


T = TypeVar("T", bound="ERTKConfig")


def resolve_files(key):
    return key


def resolve_files_load(key):
    return OmegaConf.load(key)


OmegaConf.register_resolver("file", resolve_files)
OmegaConf.register_resolver("cwdpath", resolve_files_load)


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
    def from_config(cls: Type[T], config: Any) -> T:
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
        schema = OmegaConf.structured(cls)
        return cast(T, OmegaConf.merge(schema, config))

    @classmethod
    def from_file(
        cls: Type[T], path: PathOrStr, override: Optional[List[str]] = None
    ) -> T:
        """Create config from YAML file and optionlly override some
        values.

        Parameters
        ----------
        path: os.Pathlike or str
            The path to YAML file containing config.
        override: list of str, optional
            Argument overrides in the form of key=value pairs.

        Returns
        -------
        ERKConfig
            The resulting config.
        """
        config = OmegaConf.load(Path(path))
        if override is not None:
            config = OmegaConf.merge(config, OmegaConf.from_dotlist(override))
        return cls.from_config(config)

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

    def merge_with_args(self: T, args: Optional[List[str]] = None) -> T:
        """Merge config with command-line arguments.

        Parameters
        ----------
        args: list of str, optional
            Argument overrides in the form of key=value pairs.

        Returns
        -------
        ERKConfig
            The resulting config.
        """
        return cast(T, OmegaConf.merge(self, OmegaConf.from_cli(args)))


def get_arg_mapping(s: Union[Path, str]) -> Dict[str, Any]:
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
    mapping: dict
        A dictionary mapping keys to values from the string.
    """
    if isinstance(s, Path) or Path(s).exists():
        with open(s) as fid:
            return yaml.safe_load(fid) or {}
    mapping: Dict[str, List[str]] = defaultdict(list)
    for cls in s.split(","):
        key, val = cls.split(":")
        mapping[key].append(val)
    return {k: v[0] if len(v) == 1 else v for k, v in mapping.items()}
