"""Process the raw EmoryNLP dataset.

This assumes the following file structure from the original compressed files:
/.../
    emorynlp_dev_splits/
        *.mp4
    emorynlp_test_splits/
        *.mp4
    emorynlp_train_splits/
        *.mp4
    *.csv
"""

from pathlib import Path

import click
import pandas as pd

from ertk.dataset import resample_audio, write_annotations, write_filelist
from ertk.utils import PathlibPath

NAME_FMT = "sea{0.Season}_ep{0.Episode}_sc{0.Scene_ID}_utt{0.Utterance_ID}"

emotion_map = {
    "Joyful": "joy",
    "Mad": "anger",
    "Neutral": "neutral",
    "Peaceful": "peace",
    "Powerful": "power",
    "Sad": "sadness",
    "Scared": "fear",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the EmoryNLP dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """
    splits = ["train", "dev", "test"]
    csvs = [f"emorynlp_{x}_final.csv" for x in splits]

    paths = list(input_dir.glob("**/*.mp4"))
    resample_dir = Path("resampled")
    if resample:
        resample_audio(paths, resample_dir)

    dfs = []
    for split, x in zip(splits, csvs):
        df = pd.read_csv(input_dir / x)
        df.index = df.apply(lambda x: NAME_FMT.format(x), axis=1)
        df["split"] = split
        write_filelist({resample_dir / f"{x}.wav" for x in df.index}, f"files_{split}")
        dfs.append(df)
    df = pd.concat(dfs)
    df["Emotion"] = df["Emotion"].map(emotion_map)

    write_filelist({resample_dir / f"{x}.wav" for x in df.index}, "files_all")
    write_annotations(df["Emotion"].to_dict(), "label")
    write_annotations(df["Speaker"].to_dict(), "speaker")
    write_annotations(df["split"].to_dict(), "split")
    write_annotations({x: "us" for x in df.index}, "country")
    write_annotations({x: "en" for x in df.index}, "language")


if __name__ == "__main__":
    main()
