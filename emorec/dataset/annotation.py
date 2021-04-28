from typing import Dict, Mapping, Optional, Type, TypeVar, Union

import pandas as pd

from ..utils import PathOrStr

AT = TypeVar("AT", str, int, float)


def read_annotations(
    filename: PathOrStr, dtype: Optional[Type[AT]] = None
) -> Dict[str, AT]:
    """Returns a dict of the form {name: annotation}."""
    # Need index_col to be False or None due to
    # https://github.com/pandas-dev/pandas/issues/9435
    df = pd.read_csv(filename, index_col=False, header=0, converters={0: str, 1: dtype})
    type_ = df.columns[1]
    annotations = df.set_index("name")[type_].to_dict()
    return annotations


def write_annotations(
    annotations: Mapping[str, object],
    name: str = "label",
    path: Union[PathOrStr, None] = None,
):
    """Write sorted annotations CSV.

    Args:
    -----
    annotations: mapping
        A mapping of the form {name: annotation}.
    name: str
        Name of the annotation.
    path: pathlike or str, optional
        Path to write CSV. If None, filename is name.csv
    """
    df = pd.DataFrame.from_dict(annotations, orient="index", columns=[name])
    df.index.name = "name"
    path = path or f"{name}.csv"
    df.sort_index().to_csv(path, header=True, index=True)
    print(f"Wrote CSV to {path}")
