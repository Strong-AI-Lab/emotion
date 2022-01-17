"""Process the raw ASED dataset.

This assumes the file structure from the original dataset:
/.../
    01Neutral/
        *.wav
    02Fearful/
        *.wav
"""

import shutil
from pathlib import Path

import click
from joblib import delayed

from ertk.dataset import write_annotations, write_filelist
from ertk.utils import PathlibPath, TqdmParallel

emotion_map = {
    "a": "anger",
    "h": "happiness",
    "n": "neutral",
    "f": "fear",
    "s": "sadness",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the ASED dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("**/*.wav"))
    resample_dir = Path("resampled")
    if resample:
        resample_dir.mkdir(exist_ok=True, parents=True)
        # S4-04-02-01-29 has an upper-case S instead of lower case
        TqdmParallel(len(paths), "Copying files")(
            delayed(shutil.copyfile)(path, resample_dir / (path.stem.lower() + ".wav"))
            for path in paths
        )
    paths = list(resample_dir.glob("*.wav"))
    write_filelist(paths, "files_all")

    write_annotations({p.stem: emotion_map[p.stem[0].lower()] for p in paths}, "label")
    write_annotations(
        {p.stem: ["F", "M"][int(p.stem[9:11]) - 1] for p in paths}, "gender"
    )
    write_annotations({p.stem: p.stem[3:5] for p in paths}, "sentence")
    write_annotations({p.stem: p.stem[6:8] for p in paths}, "repetition")
    write_annotations({p.stem: p.stem[-2:] for p in paths}, "speaker")
    write_annotations({p.stem: "am" for p in paths}, "language")


if __name__ == "__main__":
    main()
