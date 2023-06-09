"""Process the raw MELD dataset.

This assumes the following file structure from the original compressed files:
/.../
    dev_splits_complete/
        *.mp4
    train_splits/
        *.mp4
    output_repeated_splits_test/
        *.mp4
    *.csv
    ...
"""

from pathlib import Path

import click
import pandas as pd

from ertk.dataset import resample_rename_clips, write_annotations, write_filelist


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the MELD dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """
    splits = ["train", "dev", "test"]
    split_dirs = ["train_splits", "dev_splits_complete", "output_repeated_splits_test"]
    csvs = ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]

    resample_dir = Path("resampled")

    name_map = {}
    all_paths = []
    for split, x in zip(splits, split_dirs):
        paths = list(input_dir.glob(f"{x}/dia*.mp4"))
        all_paths.extend(paths)
        name_map.update({x: resample_dir / f"{split}_{x.stem}.wav" for x in paths})

    if resample:
        resample_rename_clips(mapping=name_map)

    dfs = []
    for split, x in zip(splits, csvs):
        df = pd.read_csv(input_dir / x)
        df.index = df.apply(
            lambda x: f"{split}_dia{x['Dialogue_ID']}_utt{x['Utterance_ID']}", axis=1
        )
        if split == "train":
            # Train dia125_utt3 is broken
            df = df.drop("train_dia125_utt3")
        df["split"] = split
        write_filelist({resample_dir / f"{x}.wav" for x in df.index}, f"files_{split}")
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.drop(index={"dev_dia110_utt7"})
    write_filelist({resample_dir / f"{x}.wav" for x in df.index}, "files_all")

    write_annotations(df["Emotion"], "label")
    write_annotations(df["Speaker"], "speaker")
    write_annotations(df["split"], "split")
    write_annotations(df["Utterance"], "transcript")
    write_annotations({x: "us" for x in df.index}, "country")
    write_annotations({x: "en" for x in df.index}, "language")


if __name__ == "__main__":
    main()
