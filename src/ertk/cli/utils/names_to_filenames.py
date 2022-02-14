from pathlib import Path

import click
import pandas as pd

from ertk.dataset import get_audio_paths
from ertk.utils import PathlibPath


@click.command()
@click.argument("clips", type=PathlibPath(exists=True))
@click.argument("filelist", type=PathlibPath(exists=True))
def main(clips: Path, filelist: Path):
    """Convert names in CLIPS to filepaths from FILELIST. Write to
    stdout.
    """

    if clips.suffix == ".csv":
        df = pd.read_csv(clips, header=0, index_col=0)
        names = list(df.index)
    else:
        with open(clips) as fid:
            names = list(map(str.strip, fid))

    paths = get_audio_paths(filelist)
    name_to_file = {p.stem: str(p) for p in paths}

    print("\n".join(map(name_to_file.__getitem__, names)))


if __name__ == "__main__":
    main()
