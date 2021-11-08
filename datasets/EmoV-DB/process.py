"""Process the raw EMOVO dataset.

This assumes the file structure from the original compressed file:
/.../
    f1/
        *.wav
    m1/
        *.wav
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist
from ertk.utils import PathlibPath

emotion_map = {
    "Amused": "amusement",
    "Angry": "anger",
    "Disgusted": "disgust",
    "Neutral": "neutral",
    "Sleepy": "sleepiness",
}

gender_map = {"bea": "F", "jenie": "F", "josh": "M", "sam": "M"}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the EMOVO dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """
    paths = list(input_dir.glob("**/*.wav"))
    mapping = {x: f"{x.parts[-3]}_{x.stem.lower()}" for x in paths}
    resample_dir = Path("resampled")
    if resample:
        resample_audio(paths, resample_dir, mapping=mapping)
    resampled_paths = list(resample_dir.glob("*.wav"))
    write_filelist(resampled_paths, "files_all")

    write_annotations({mapping[p]: emotion_map[p.parts[-2]] for p in paths}, "label")
    write_annotations({mapping[p]: p.parts[-3] for p in paths}, "speaker")
    write_annotations({mapping[p]: gender_map[p.parts[-3]] for p in paths}, "gender")
    write_annotations({mapping[p]: p.stem[-4:] for p in paths}, "sentence")
    write_annotations({mapping[p]: "us" for p in paths}, "country")
    write_annotations({mapping[p]: "en" for p in paths}, "language")


if __name__ == "__main__":
    main()
