"""Process the raw ESD dataset.

This assumes the file structure from the original compressed file:
/.../
    ReadMe.txt
    0001/
        0001.txt
        Angry/
            train/
            evaluation/
            test/
        Happy/
            train/
            evaluation/
            test/
        ...
    0002/
    ...
"""

import shutil
from pathlib import Path

import click
from joblib import delayed

from ertk.dataset import write_annotations, write_filelist
from ertk.utils import PathlibPath, TqdmParallel


emotion_map = {
    "Angry": "anger",
    "Happy": "happiness",
    "Neutral": "neutral",
    "Sad": "sadness",
    "Surprise": "surprise",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the ESD dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("**/*.wav"))
    resample_dir = Path("resampled")
    if resample:
        resample_dir.mkdir(exist_ok=True)
        TqdmParallel(len(paths), "Copying files")(
            delayed(shutil.copyfile)(x, resample_dir / x.name) for x in paths
        )
    write_filelist(resample_dir.glob("*.wav"), "files_all")
    for subset in ["train", "evaluation", "test"]:
        write_filelist(
            (resample_dir / x.name for x in input_dir.glob(f"*/*/{subset}/*.wav")),
            f"files_{subset}",
        )

    labels = {}
    for emo1, emo2 in emotion_map.items():
        labels.update({x.name: emo2 for x in input_dir.glob(f"*/{emo1}/*/*.wav")})
    write_annotations(labels, "label")

    speaker_dict = {p.stem: p.stem[:4] for p in paths}
    write_annotations(speaker_dict, "speaker")
    male_speakers = {4, 5, 6, 8, 10, 11, 12, 13, 14, 20}
    write_annotations(
        {k: "M" if int(v) in male_speakers else "F" for k, v in speaker_dict.items()},
        "gender",
    )
    write_annotations(
        {k: "zh" if int(v) < 11 else "en" for k, v in speaker_dict.items()}, "language"
    )


if __name__ == "__main__":
    main()
