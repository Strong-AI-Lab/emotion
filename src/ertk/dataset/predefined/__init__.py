"""Pre-defined dataset standardisation scripts."""

import importlib
import os
from pathlib import Path
from typing import Callable, List

import importlib_resources

from ertk.utils import PathOrStr

__all__ = ["process_dataset", "list_predefined_datasets"]


def list_predefined_datasets() -> List[str]:
    """List all predefined datasets.

    Returns
    -------
    List[str]
        A list of all predefined datasets by name.
    """
    return [
        x.name
        for x in Path(__file__).parent.iterdir()
        if x.is_dir() and not x.name.startswith("_")
    ]


# TODO: refactor datasets into separate processing functions and commands
def _get_process_func(name: str) -> Callable:
    """Get a dataset class by name."""

    all_names = list_predefined_datasets()
    if name not in all_names:
        raise ValueError(f"Dataset name must be one of {all_names}")
    module = importlib.import_module(f"ertk.dataset.predefined.{name}.process")
    return module.main


def process_dataset(
    name: str, input_dir: PathOrStr, output_dir: PathOrStr, resample: bool = True
) -> None:
    """Process a pre-defined dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.
    input_dir : PathOrStr
        Path to the input directory.
    output_dir : PathOrStr
        Path to the output directory.
    resample : bool, optional
        Whether to resample the audio to 16 kHz, by default True

    Raises
    ------
    ValueError
        If the dataset name is not recognised.
    """
    func = _get_process_func(name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    func([str(input_dir)] + ["--resample"] if resample else [], standalone_mode=False)
    conf = (
        importlib_resources.files(f"ertk.dataset.predefined.{name}")
        .joinpath("corpus.yaml")
        .read_text()
    )
    with open("corpus.yaml", "w") as fid:
        fid.write(conf)
