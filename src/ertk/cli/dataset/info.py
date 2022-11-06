import logging
from pathlib import Path
from typing import Tuple

import click

from ertk.cli._utils import dataset_args, debug_args
from ertk.config import get_arg_mapping
from ertk.dataset import DataLoadConfig, load_datasets_config, load_multiple


def write_list(dataset, output_list):
    with open(output_list, "w") as fid:
        fid.write("\n".join(dataset.get_audio_paths()))


@click.command()
@dataset_args
@debug_args
@click.option("--output_list", type=Path)
def main(
    corpus_info: Tuple[Path],
    data_config: Path,
    subset: Tuple[str],
    sel_groups: Tuple[str],
    remove_groups: Tuple[str],
    map_groups: Tuple[str],
    clip_seq: int,
    pad_seq: int,
    verbose: int,
    output_list: Path,
):
    """Print info about a dataset or combination of datasets."""
    if verbose > 0:
        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)

    if data_config:
        dataset = load_datasets_config(DataLoadConfig.from_file(data_config))
        print(dataset)
        if output_list:
            write_list(dataset, output_list)
        return

    if len(subset) == 1 and ":" not in subset[0]:
        dataset = load_multiple(corpus_info, subsets=subset[0], label="")
    else:
        subset_map = {}
        for m in subset:
            subset_map.update(get_arg_mapping(m))
        dataset = load_multiple(corpus_info, subsets=subset_map, label="")

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
    if output_list:
        write_list(dataset, output_list)


if __name__ == "__main__":
    main()
