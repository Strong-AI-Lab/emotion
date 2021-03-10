import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import click
from emorec.dataset import get_audio_paths, parse_annotations
from emorec.utils import PathlibPath


@click.command()
@click.argument('input', type=PathlibPath(exists=True, dir_okay=False))
@click.argument('labels', type=PathlibPath(exists=True, dir_okay=False))
@click.argument('speakers', type=PathlibPath(exists=True, dir_okay=False))
@click.argument('output', type=Path)
def main(input: Path, labels: Path, speakers: Path, output: Path):
    """Create directory structure with speaker-independent
    cross-validation folds. Each speaker has a directory which is
    patitioned by label.
    """

    spk_dict = parse_annotations(speakers, dtype=str)
    paths = get_audio_paths(input)
    speaker_paths: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        speaker_paths[spk_dict[path.stem]].append(path)

    lbl_dict = parse_annotations(labels)
    for i, speaker in enumerate(speaker_paths.keys()):
        for path in speaker_paths[speaker]:
            emotion = lbl_dict[path.stem]
            fold = f'fold_{i + 1:d}'
            newpath = output / fold / emotion / path.name
            newpath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(path), str(newpath))
            print(newpath)


if __name__ == "__main__":
    main()
