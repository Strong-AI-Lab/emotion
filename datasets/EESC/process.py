"""Process the raw EESC dataset.

This assumes the file structure from the original compressed file:
/.../
    *.wav
    *.TextGrid
"""

from pathlib import Path

import click
import textgrid
from tqdm import tqdm

from ertk.dataset import resample_audio, write_annotations, write_filelist


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the EESC dataset at location INPUT_DIR and optionally
    resample audio.
    """
    paths = list(input_dir.glob("*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_dir.mkdir(exist_ok=True)
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all")
    else:
        write_filelist(paths, "files_all")

    labels = {}
    sentences = {}
    for path in tqdm(
        input_dir.glob("*.TextGrid"), desc="Processing annotations", total=len(paths)
    ):
        grid = textgrid.TextGrid.fromFile(path)
        labels[path.stem] = grid.getFirst("emotion")[0].mark
        sentences[path.stem] = grid.getFirst("sentence")[0].mark
    write_annotations(labels, "label")
    write_annotations(sentences, "transcript")
    write_annotations({p.stem: "et" for p in paths}, "language")
    write_annotations({p.stem: "ee" for p in paths}, "country")


if __name__ == "__main__":
    main()
