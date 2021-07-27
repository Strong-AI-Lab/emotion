"""Process the raw CaFE dataset.

This assumes the file structure from the original compressed file:
/.../
    Colère/
        Faible/
            *.aiff [for 192k data]
            *.wav  [for 48k data]
        Fort/
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist
from ertk.utils import PathlibPath

emotion_map = {
    "C": "anger",
    "D": "disgust",
    "J": "happiness",
    "N": "neutral",
    "P": "fear",
    "S": "surprise",
    "T": "sadness",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the CaFE dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("**/*.wav"))
    if len(paths) == 0:
        paths = list(input_dir.glob("**/*.aiff"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"))

    write_annotations({p.stem: emotion_map[p.stem[3]] for p in paths}, "label")
    speaker_dict = {p.stem: p.stem[:2] for p in paths}
    write_annotations(speaker_dict, "speaker")
    write_annotations(
        {k: ["F", "M"][int(v) % 2] for k, v in speaker_dict.items()},
        "gender",
    )
    write_annotations({p.stem: "fr" for p in paths}, "language")


if __name__ == "__main__":
    main()
