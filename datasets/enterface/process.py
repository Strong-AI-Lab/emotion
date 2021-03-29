"""Process the raw eNTERFACE dataset.

This assumes the file structure from the original compressed file:
/.../
    subject 1/
        anger/
            sentence 1/
                *.avi
            ...
        ...
    ...
"""

import shutil
from pathlib import Path

import click
from emorec.dataset import resample_audio, write_filelist, write_annotations
from emorec.utils import PathlibPath

emotion_map = {
    "an": "anger",
    "di": "disgust",
    "fe": "fear",
    "ha": "happiness",
    "sa": "sadness",
    "su": "surprise",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the eNTERFACE dataset at location INPUT_DIR and convert
    AVI to 16 kHz 16-bit WAV audio.
    """
    paths = list(input_dir.glob("**/*.avi"))
    resample_dir = Path("resampled")

    # Correct duplicate names for subject 11 separately
    sub11 = [p for p in paths if "subject 11" in p.parts]
    resample_audio(sub11, resample_dir)
    for f in resample_dir.glob("s12*"):
        shutil.move(f, f.with_name(f.name.replace("s12", "s11")))

    rest = [p for p in paths if "subject 11" not in p.parts]
    resample_audio(rest, resample_dir)

    # More manual corrections
    shutil.move("resampled/s16_su_3avi.wav", "resampled/s16_su_3.wav")
    for f in resample_dir.glob("s_3_*"):
        shutil.move(f, f.with_name(f.name.replace("s_3", "s3")))

    newpaths = list(
        filter(lambda p: not p.stem.startswith("s6_"), resample_dir.glob("*.wav"))
    )
    write_filelist(newpaths)
    write_annotations({p.stem: emotion_map[p.stem[-4:-2]] for p in newpaths})
    write_annotations({p.stem: p.stem[: p.stem.find("_")] for p in newpaths}, "speaker")


if __name__ == "__main__":
    main()
