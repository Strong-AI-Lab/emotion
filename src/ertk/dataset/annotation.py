from typing import Mapping, Optional, Type, Union

import pandas as pd
from typing_extensions import Literal

from ertk.utils import PathOrStr


def read_annotations(
    filename: PathOrStr, dtype: Optional[Union[Type, Literal["category"]]] = None
) -> pd.Series:
    """Returns a pd.Series containing values for this annotation for
    each instance, indexed by name.
    """
    # Need index_col to be False or None due to
    # https://github.com/pandas-dev/pandas/issues/9435
    _dtype = dtype
    if dtype == "category":
        _dtype = str
    df = pd.read_csv(
        filename,
        index_col=0,
        header=0,
        converters={0: str, 1: _dtype},
        low_memory=False,
    )
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
        df = df[df.columns[0]]
    elif isinstance(annotations, pd.Series):
        df = annotations
    else:
        if not name:
            raise ValueError("`name` must be given when passing a dict.")
        df = pd.DataFrame.from_dict(annotations, orient="index", columns=[name])
        df = df[name]
    df.index.name = "name"
    name = name or df.name
    path = path or f"{name}.csv"
    df.sort_index().to_csv(path, header=True, index=True)
