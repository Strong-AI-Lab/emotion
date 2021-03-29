"""Process the raw SAVEE dataset.

This assumes the file structure from the original compressed file:
/.../
    DC/
        *.wav
    ...
"""

import re
import shutil
from pathlib import Path

import click
from emorec.dataset import resample_audio, write_filelist, write_annotations
from emorec.utils import PathlibPath

REGEX = re.compile(r"^(DC|JE|JK|KL)([a-z][a-z]?)[0-9][0-9]$")

emotion_map = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happiness",
    "n": "neutral",
    "sa": "sadness",
    "su": "surprise",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the SAVEE dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    resample_dir = Path("resampled")
    for sp in ["DC", "JE", "JK", "KL"]:
        paths = list(input_dir.glob(sp + "/*.wav"))
        resample_audio(paths, resample_dir / sp)
    for sp in ["DC", "JE", "JK", "KL"]:
        for f in (resample_dir / sp).glob("*.wav"):
            shutil.move(f, resample_dir / (sp + f.name))
        (resample_dir / sp).rmdir()

    paths = list(resample_dir.glob("*.wav"))
    write_filelist(paths)
    write_annotations(
        {p.stem: emotion_map[REGEX.match(p.stem).group(2)] for p in paths}
    )
    write_annotations({p.stem: p.stem[:2] for p in paths}, "speaker")


if __name__ == "__main__":
    main()
