"""Process the raw OréauFR dataset.

This assumes the file structure from the original compressed file:
/.../
    OréauFR_01/
        m/
            sessc/
                *.wav
            ...
        f/
        contents.txt
    OréauFR_02/
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist

emotion_map = {
    "C": "anger",
    "D": "disgust",
    "J": "happiness",
    "N": "neutral",
    "P": "fear",
    "S": "surprise",
    "T": "sadness",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process OréauFR dataset at location INPUT_DIR."""

    paths = list(input_dir.glob("OréauFR_01/*/*/*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_dir.mkdir(exist_ok=True)
        resample_audio(paths, resample_dir)
        paths = list(resample_dir.glob("*.wav"))
    write_filelist(paths, "files_all")
    write_filelist(paths, "files_01")
    write_filelist(input_dir.glob("OréauFR_02/*/*/*.wav"), "files_02")

    write_annotations({p.stem: emotion_map[p.stem[5]] for p in paths}, "label")
    write_annotations({p.stem: p.stem[0:2] for p in paths}, "speaker")
    write_annotations({p.stem: p.stem[2:5] for p in paths}, "sentence")
    write_annotations({p.stem: "fr" for p in paths}, "language")


if __name__ == "__main__":
    main()
