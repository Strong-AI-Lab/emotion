"""Process the raw EmoFilm dataset.

This assumes the file structure from the original compressed file:
/.../
    *.wav
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist
from ertk.utils import PathlibPath

emotion_map = {
    "ans": "fear",
    "dis": "disgust",
    "gio": "happiness",
    "rab": "anger",
    "tri": "sadness",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the EmoFilm dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """
    paths = list(input_dir.glob("*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"))

    write_annotations({p.stem: emotion_map[p.stem[2:5]] for p in paths}, "label")
    write_annotations({p.stem: p.stem[-2:] for p in paths}, "speaker")
    # Speaker is language in this case
    write_annotations({p.stem: p.stem[-2:] for p in paths}, "language")


if __name__ == "__main__":
    main()
