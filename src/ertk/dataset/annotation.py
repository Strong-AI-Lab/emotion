"""Functions for reading and writing annotations."""

from collections.abc import Mapping
from typing import Literal, Optional, Union

import pandas as pd

from ertk.utils import PathOrStr

__all__ = ["read_annotations", "write_annotations"]


def read_annotations(
    filename: PathOrStr, dtype: Optional[Union[type, Literal["category"]]] = None
) -> pd.Series:
    """Returns a pd.Series containing values for this annotation for
    each instance, indexed by name.
    """
    _dtype = dtype
    if _dtype == "category":
        _dtype = str
    dtypes: dict[int, type] = {0: str}
    if _dtype is not None:
        dtypes[1] = _dtype
    df = pd.read_csv(filename, index_col=0, header=0, dtype=dtypes, low_memory=False)
    if dtype == "category":
        return df[df.columns[0]].astype("category")
    return df[df.columns[0]]


def write_annotations(
    annotations: Union[Mapping[str, object], pd.DataFrame, pd.Series],
    name: Optional[str] = None,
    path: Optional[PathOrStr] = None,
) -> None:
    """Write sorted annotations CSV.

    Parameters
    ----------
    annotations: mapping
        A mapping of the form {name: annotation}.
    name: str, optional
        Name of the annotation if passing a dict.
    path: pathlike or str, optional
        Path to write CSV. If None, filename is name.csv
    """
    if isinstance(annotations, pd.DataFrame):
        df = annotations
        if len(df.columns) == 2:
            try:
                df.set_index("name", inplace=True)
            except KeyError:
                raise ValueError("Passed DataFrame should have 'name' as index.")
        elif len(df.columns) > 2:
            raise ValueError("Passed DataFrame should have 1 or 2 columns.")
        series = df[df.columns[0]]
    elif isinstance(annotations, pd.Series):
        series = annotations
    else:
        if not name:
            raise ValueError("`name` must be given when passing a dict.")
        df = pd.DataFrame.from_dict(annotations, orient="index", columns=[name])
        series = df[name]
    series.index.name = "name"
    name = name or series.name
    series.name = name
    path = path or f"{name}.csv"
    series.sort_index().to_csv(path, header=True, index=True)
