"""Process the raw Portuguese dataset (Castro & Lima).

This assumes the file structure from the original compressed file:
/.../
    *.wav
"""

import re
from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist
from ertk.utils import PathlibPath

REGEX = re.compile(r"^\d+[sp][AB]_([a-z]+)\d+$")

emotion_map = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "sad": "sadness",
    "neutral": "neutral",
    "surprise": "surprise",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the Portuguese dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all")

    write_annotations(
        {p.stem: emotion_map[REGEX.match(p.stem).group(1)] for p in paths},
        "label",
    )
    write_annotations({p.stem: p.stem[p.stem.find("_") - 1] for p in paths}, "speaker")
    write_annotations({p.stem: "pt" for p in paths}, "language")


if __name__ == "__main__":
    main()
