import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import click

from ertk.dataset import Dataset, get_audio_paths


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=Path)
@click.option(
    "--label", "label_name", default="label", help="Categorical label to use."
)
def main(input: Path, output: Path, label_name: str):
    """Create directory structure with speaker-independent
    cross-validation folds. Each speaker has a directory which is
    patitioned by label.
    """

    dataset = Dataset(input)
    paths = get_audio_paths(dataset._subset_paths[dataset.subset])
    speaker_paths: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        speaker_paths[dataset.annotations["speaker"][path.stem]].append(path)

    for i, speaker in enumerate(speaker_paths.keys()):
        for path in speaker_paths[speaker]:
            label = dataset.annotations[label_name][path.stem]
            fold = f"fold_{i + 1:d}"
            newpath = output / fold / label / path.name
            newpath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(path), str(newpath))
            print(newpath)


if __name__ == "__main__":
    main()
