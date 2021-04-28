"""Process the raw ShEMO dataset.

This assumes the file structure from the original compressed file:
/.../
    male/
        *.wav
    female/
    ...
"""

from pathlib import Path

import click

from emorec.dataset import resample_audio, write_annotations, write_filelist
from emorec.utils import PathlibPath

emotion_map = {
    "A": "anger",
    "H": "happiness",
    "N": "neutral",
    "S": "sadness",
    "W": "surprise",
    "F": "fear",
}

unused_emotions = ["F"]


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the ShEMO dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("*/*.wav"))
    resample_dir = Path("resampled")
    resample_audio(paths, resample_dir)

    write_filelist(
        [p for p in resample_dir.glob("*.wav") if p.stem[3] not in unused_emotions]
    )
    write_annotations({p.stem: emotion_map[p.stem[3]] for p in paths})
    speaker_dict = {p.stem: p.stem[:3] for p in paths}
    write_annotations(speaker_dict, "speaker")
    male_speakers = [f"M{i:02d}" for i in range(1, 57)]
    gender_dict = {
        k: "M" if v in male_speakers else "F" for k, v in speaker_dict.items()
    }
    write_annotations(gender_dict, "gender")


if __name__ == "__main__":
    main()
