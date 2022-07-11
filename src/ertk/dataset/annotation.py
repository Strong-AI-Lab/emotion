from typing import Dict, Mapping, Optional, Type, TypeVar, Union

import pandas as pd

from ertk.utils import PathOrStr

AT = TypeVar("AT", str, int, float)


def read_annotations(
    filename: PathOrStr, dtype: Optional[Type[AT]] = None
) -> Dict[str, AT]:
    """Returns a dict of the form {name: annotation}."""
    # Need index_col to be False or None due to
    # https://github.com/pandas-dev/pandas/issues/9435
    df = pd.read_csv(
        filename,
        index_col=False,
        header=0,
        converters={0: str, 1: dtype},
        low_memory=False,
    )
    annotations = df.set_index(df.columns[0])[df.columns[1]].to_dict()
    return annotations


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
