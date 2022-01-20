from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def get_arg_mapping_multi(s: str) -> Dict[str, List[Any]]:
    """Given a string mapping from the command-line, returns a dict
    representing that mapping.

    The string form of the mapping is:
        key:value[,key:value]+
    Duplicate keys will be mapped to a list of values.

    Args:
    -----
    s: str
        String representing the mapping. It cannot contain spaces or
        shell symbols (unless escaped).

    Returns:
    --------
    mapping: dict
        A dictionary mapping keys to lists of values from the string.
    """
    mapping: Dict[str, List[str]] = {}
    for cls in s.split(","):
        key, val = cls.split(":")
        if key in mapping:
            mapping[key].append(val)
        else:
            mapping[key] = [val]
    return mapping


def get_arg_mapping(s: Union[Path, str]) -> Dict[str, Any]:
    """Given a mapping on the command-line, returns a dict representing
    that mapping. Mapping can be a string or a more complex YAML file.

    The string form of the mapping is:
        key:value[,key:value]+

    Args:
    -----
    s: PathLike or str
        String representing the mapping or path to YAML containing
        mapping. If a string, it cannot contain spaces or shell symbols
        (unless escaped).

    Returns:
    --------
    mapping: dict
        A dictionary mapping keys to values from the string.
    """
    if isinstance(s, Path) or Path(s).exists():
        with open(s) as fid:
            return yaml.safe_load(fid) or {}
    return {k: v[0] if len(v) == 1 else v for k, v in get_arg_mapping_multi(s).items()}
