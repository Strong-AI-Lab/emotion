import shutil
from collections import defaultdict
from pathlib import Path

import click

from ertk.dataset import Dataset, get_audio_paths


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=Path)
@click.option(
    "--label", "label_name", default="label", help="Categorical label to use."
)
@click.option("--partition", default="speaker", help="Partition to split on.")
def main(input: Path, output: Path, label_name: str, partition: str):
    """Create directory structure with group-independent
    cross-validation folds. Each group has a directory which is
    patitioned by label.
    """

    dataset = Dataset(input)
    paths = get_audio_paths(dataset._subset_paths[dataset.subset])
    group_paths: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        group_paths[dataset.annotations.loc[path.stem, partition]].append(path)

    for i, group in enumerate(group_paths.keys()):
        for path in group_paths[group]:
            label = dataset.annotations.loc[path.stem, label_name]
            fold = f"fold_{i + 1:d}"
            newpath = output / fold / label / path.name
            newpath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(path), str(newpath))
            print(newpath)


if __name__ == "__main__":
    main()
