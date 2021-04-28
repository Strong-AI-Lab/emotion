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

from emorec.dataset import resample_audio, write_annotations, write_filelist
from emorec.utils import PathlibPath

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
def main(input_dir: Path):
    """Process the RAVDESS dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("Audio/Actor_??/03-01-*.wav"))
    resample_dir = Path("resampled")
    resample_audio(paths, resample_dir)

    write_filelist(resample_dir.glob("*.wav"))
    write_annotations({p.stem: emotion_map[p.stem[6:8]] for p in paths})
    speaker_dict = {p.stem: p.stem[-2:] for p in paths}
    write_annotations(speaker_dict, "speaker")
    male_speakers = [f"{i:02d}" for i in range(1, 25, 2)]
    gender_dict = {
        k: "M" if v in male_speakers else "F" for k, v in speaker_dict.items()
    }
    write_annotations(gender_dict, "gender")


if __name__ == "__main__":
    main()
