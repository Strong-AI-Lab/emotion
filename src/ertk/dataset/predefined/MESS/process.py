"""Process the raw MESS dataset.

This assumes the file structure from the original dataset:
/.../
    AF1M01_SCR.wav
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist

emotion_map = {
    "A": "anger",
    "C": "calm",
    "H": "happiness",
    "S": "sadness",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the MESD dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all")
    else:
        write_filelist(paths, "files_all")

    write_annotations({p.stem: emotion_map[p.stem[0]] for p in paths}, "label")
    write_annotations({p.stem: p.stem[1] for p in paths}, "gender")
    write_annotations({p.stem: p.stem[1:3] for p in paths}, "speaker")
    write_annotations({p.stem: p.stem[3] for p in paths}, "cue")
    write_annotations({p.stem: "en" for p in paths}, "language")
    write_annotations({p.stem: "us" for p in paths}, "country")


if __name__ == "__main__":
    main()
