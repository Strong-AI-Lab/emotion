"""Process the raw MESD dataset.

This assumes the file structure from the original dataset:
/.../
    Anger_C_A_abajo.wav
    Anger_C_A_adios.wav
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist


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
    resample_dir = Path("resampled")
    if resample:
        resample_audio(paths, resample_dir)
    write_filelist(resample_dir.glob("*.wav"), "files_all")

    write_annotations({p.stem: p.stem.split("_")[0].lower() for p in paths}, "label")
    write_annotations({p.stem: p.stem.split("_", maxsplit=3)[3] for p in paths}, "word")
    write_annotations({p.stem: p.stem.split("_")[1] for p in paths}, "voice")
    write_annotations({p.stem: p.stem.split("_")[2] for p in paths}, "corpus")
    write_annotations({p.stem: "es" for p in paths}, "language")
    write_annotations({p.stem: "mx" for p in paths}, "country")


if __name__ == "__main__":
    main()
