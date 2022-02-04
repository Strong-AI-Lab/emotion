"""Process the raw DEMoS dataset.

This assumes the file structure from the original compressed file:
/.../
    DEMOS/
        NP_*.wav
        PR_*.wav
    NEU/
        *.wav
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist
from ertk.utils import PathlibPath

emotion_map = {
    "rab": "anger",
    "tri": "sadness",
    "gio": "happiness",
    "pau": "fear",
    "dis": "disgust",
    "col": "guilt",
    "sor": "surprise",
    "neu": "neutral",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the DEMoS dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """
    paths = list(input_dir.glob("**/*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all")
        write_filelist(resample_dir.glob("PR_*.wav"), "files_PR")

    write_annotations({p.stem: emotion_map[p.stem[-6:-3]] for p in paths}, "label")
    speaker_dict = {p.stem: p.stem[-9:-7] for p in paths}
    write_annotations(speaker_dict, "speaker")
    # fmt: off
    male_speakers = [
        "02", "03", "04", "05", "08", "09", "10", "11", "12", "14", "15", "16", "18",
        "19", "23", "24", "25", "26", "27", "28", "30", "33", "34", "39", "41", "50",
        "51", "52", "53", "58", "59", "63", "64", "65", "66", "67", "68", "69"
    ]
    # fmt: on
    gender_dict = {
        k: "M" if v in male_speakers else "F" for k, v in speaker_dict.items()
    }
    write_annotations(gender_dict, "gender")
    write_annotations({p.stem: "it" for p in paths}, "language")
    write_annotations({p.stem: "it" for p in paths}, "country")


if __name__ == "__main__":
    main()
