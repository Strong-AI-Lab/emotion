"""Process the raw RAVDESS dataset.

This assumes the file structure from the original compressed file:
/.../
    Audio/
        Actor_01/
            *.wav
        ...
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist
from ertk.utils import PathlibPath

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happiness",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the RAVDESS dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("Audio/Actor_??/03-*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all.txt")
        write_filelist(resample_dir.glob("03-01-*.wav"), "files_speech.txt")
        write_filelist(resample_dir.glob("03-02-*.wav"), "files_song.txt")

    write_annotations({p.stem: emotion_map[p.stem[6:8]] for p in paths}, "label")
    speaker_dict = {p.stem: p.stem[-2:] for p in paths}
    write_annotations(speaker_dict, "speaker")
    write_annotations(
        {k: ["F", "M"][int(v) % 2] for k, v in speaker_dict.items()}, "gender"
    )
    write_annotations({p.stem: "en" for p in paths}, "language")
    write_annotations({p.stem: p.stem[9:11] for p in paths}, "intensity")
    write_annotations({p.stem: p.stem[12:14] for p in paths}, "statement")


if __name__ == "__main__":
    main()
