"""Process the raw SUBESCO dataset.

This assumes the file structure from the original compressed file:
/.../
    Colère/
        Faible/
            *.aiff [for 192k data]
            *.wav  [for 48k data]
        Fort/
    ...
"""

import re
from pathlib import Path
from typing import Dict

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist

REGEX = re.compile(r"^([MF])_([0-9]+)_([A-Z]+)_S_([0-9]+)_([A-Z]+)_([0-9])$")

emotion_map = {
    "ANGRY": "anger",
    "DISGUST": "disgust",
    "HAPPY": "happiness",
    "NEUTRAL": "neutral",
    "FEAR": "fear",
    "SURPRISE": "surprise",
    "SAD": "sadness",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the SUBESCO dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all")

    keys = ["label", "sentence", "speaker", "speaker_name", "gender"]
    annot: Dict[str, Dict[str, str]] = {x: {} for x in keys}
    for p in paths:
        match = REGEX.match(p.stem)
        if not match:
            print(p)
        else:
            annot["gender"][p.stem] = match[1]
            annot["speaker"][p.stem] = match[2]
            annot["speaker_name"][p.stem] = match[3]
            annot["sentence"][p.stem] = match[4]
            annot["label"][p.stem] = emotion_map[match[5]]

    for k in keys:
        write_annotations(annot[k], k)
    write_annotations({p.stem: "bn" for p in paths}, "language")


if __name__ == "__main__":
    main()
