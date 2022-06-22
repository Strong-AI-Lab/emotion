from pathlib import Path
from typing import Tuple

import click

from ertk.cli._utils import dataset_args
from ertk.config import get_arg_mapping
from ertk.dataset import load_multiple


@click.command()
@dataset_args
def main(
    corpus_info: Tuple[Path],
    subset: Tuple[str],
    sel_groups: Tuple[str],
    remove_groups: Tuple[str],
    map_groups: Tuple[str],
    clip_seq: int,
    pad_seq: int,
):
    """Print info about a dataset or combination of datasets."""
    if len(subset) == 1 and ":" not in subset[0]:
        dataset = load_multiple(corpus_info, subsets=subset[0])
    else:
        subset_map = {}
        for m in subset:
            subset_map.update(get_arg_mapping(m))
        dataset = load_multiple(corpus_info, subsets=subset_map)

    grp_map = {}
    for m in map_groups:
        grp_map.update(get_arg_mapping(m))
    grp_sel = {}
    for m in sel_groups:
        grp_sel.update(get_arg_mapping(m))
    grp_del = {}
    for m in remove_groups:
        grp_del.update(get_arg_mapping(m))
    dataset.map_and_select(grp_map, grp_sel, grp_del)

    if clip_seq:
        dataset.clip_arrays(clip_seq)
    if pad_seq:
        dataset.pad_arrays(pad_seq)

    print(dataset)


if __name__ == "__main__":
    main()
