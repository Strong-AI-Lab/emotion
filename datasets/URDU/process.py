"""Process the raw URDU dataset.

This assumes the file structure from the original compressed file:
/.../
    Angry/
        *.wav
    ...
"""

from pathlib import Path

import click

from emorec.dataset import resample_audio, write_annotations, write_filelist
from emorec.utils import PathlibPath

emotion_map = {
    "A": "anger",
    "S": "sadness",
    "H": "happiness",
    "N": "neutral",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the URDU dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("*/*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"))

    write_annotations(
        {p.stem: emotion_map[p.stem[p.stem.rfind("_") + 1]] for p in paths}, "label"
    )
    write_annotations({p.stem: p.stem[: p.stem.index("_")] for p in paths}, "speaker")
    write_annotations({p.stem: "ur" for p in paths}, "language")


if __name__ == "__main__":
    main()
