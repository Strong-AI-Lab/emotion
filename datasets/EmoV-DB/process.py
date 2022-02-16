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
"""

from pathlib import Path

import click

from ertk.dataset import resample_rename_clips, write_annotations, write_filelist

emotion_map = {
    "Amused": "amusement",
    "Angry": "anger",
    "Disgusted": "disgust",
    "Neutral": "neutral",
    "Sleepy": "sleepiness",
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
    if resample:
        resample_rename_clips(mapping=mapping)
    resampled_paths = list(resample_dir.glob("*.wav"))
    write_filelist(resampled_paths, "files_all")

    write_annotations(
        {mapping[p].stem: emotion_map[p.parts[-2]] for p in paths}, "label"
    )
    write_annotations({mapping[p].stem: p.parts[-3] for p in paths}, "speaker")
    write_annotations(
        {mapping[p].stem: gender_map[p.parts[-3]] for p in paths}, "gender"
    )
    write_annotations({mapping[p].stem: p.stem[-4:] for p in paths}, "sentence")
    write_annotations({mapping[p].stem: "us" for p in paths}, "country")
    write_annotations({mapping[p].stem: "en" for p in paths}, "language")


if __name__ == "__main__":
    main()
