"""Process the raw EmoV-DB dataset.

This assumes the file structure from the original sorted data:
/.../
    bea/
        Angry/
            *.wav
        Amused/
            *.wav
        ...
    josh/
        Angry/
            *.wav
        ...
    ...
    cmuarctic.data
"""

import re
from pathlib import Path

import click

from ertk.dataset import resample_rename_clips, write_annotations, write_filelist

emotion_map = {
    "amused": "amusement",
    "anger": "anger",
    "disgust": "disgust",
    "neutral": "neutral",
    "sleepiness": "sleepiness",
}

gender_map = {"bea": "F", "jenie": "F", "josh": "M", "sam": "M"}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the EmoV-DB dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """
    paths = list(input_dir.glob("**/*.wav"))
    resample_dir = Path("resampled")
    mapping = {x: resample_dir / f"{x.parts[-3]}_{x.stem.lower()}.wav" for x in paths}
    resample_rename_clips(mapping=mapping)
    paths = list(resample_dir.glob("*.wav"))
    write_filelist(paths, "files_all")

    write_annotations(
        {p.stem: emotion_map[p.stem.split("_")[1]] for p in paths}, "label"
    )
    write_annotations({p.stem: p.stem.split("_")[0] for p in paths}, "speaker")
    write_annotations(
        {p.stem: gender_map[p.stem.split("_")[0]] for p in paths}, "gender"
    )
    write_annotations({p.stem: p.stem[-4:] for p in paths}, "sentence")
    write_annotations({p.stem: "us" for p in paths}, "country")
    write_annotations({p.stem: "en" for p in paths}, "language")

    sent_to_transcript = {}
    with open(input_dir / "cmuarctic.data") as fid:
        for line in fid:
            match = re.match(r"^\( arctic_a([0-9]{4}) \"(.*)\" \)$", line.strip())
            if match:
                sent_to_transcript[match.group(1)] = match.group(2)
    write_annotations(
        {p.stem: sent_to_transcript[p.stem[-4:]] for p in paths}, "transcript"
    )


if __name__ == "__main__":
    main()
