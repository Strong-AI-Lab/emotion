from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

from ertk.dataset import read_annotations


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "speaker_info", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--speakers", help="Speakers to select (comma-separated).")
@click.option(
    "--num_clips",
    type=int,
    default=20,
    help="Number of clips to select per speaker.",
    show_default=True,
)
@click.option(
    "--min_clips",
    type=int,
    default=0,
    help="Minimum number of clips required for speaker to be included.",
    show_default=True,
)
@click.option("--plot/--noplot", default=False, help="Plot histogram of scores.")
def main(
    input: Path,
    speaker_info: Path,
    output: Path,
    speakers: str,
    num_clips: int,
    min_clips: int,
    plot: bool,
):
    """Filters predictions from INPUT in a stratified manner according
    to the speakers in SPEAKER_INFO. For each speaker the same number of
    top/bottom clips are selected. A new CSV file is written to OUTPUT.
    """

    df = pd.read_csv(input, header=0, index_col=0)
    spk_dict = read_annotations(speaker_info, dtype=str)
    df["Speaker"] = df.index.map(spk_dict.get)
    df.sort_values("Score", inplace=True)

    print(f"Scores for {len(df)} clips:")
    print(df["Score"].describe())
    print()
    if plot:
        ax = df["Score"].plot(
            kind="hist",
            bins=200,
            xlim=(0, 1),
            xlabel="Score",
            title="Histogram of scores",
        )
        ax.vlines(
            [df["Score"].mean(), df["Score"].median()],
            ymin=0,
            ymax=ax.get_ylim()[1],
            colors=["red", "orange"],
        )
        plt.show()

    sp_cls = df.groupby(["Speaker"]).size().sort_values()
    print("Speaker counts:")
    print(sp_cls)
    print()
    if speakers is not None:
        valid_speakers = sp_cls.index.isin(speakers.split(","))
    else:
        valid_speakers = sp_cls[sp_cls > min_clips].index
    print(f"{len(valid_speakers)} valid speakers.")

    large_speakers = sp_cls[sp_cls >= num_clips].index.intersection(valid_speakers)
    small_speakers = sp_cls[sp_cls < num_clips].index.intersection(valid_speakers)
    large_df = df[df["Speaker"].isin(large_speakers)]
    small_df = df[df["Speaker"].isin(small_speakers)]
    gb = large_df.groupby("Speaker")
    out_df = pd.concat([gb.head(num_clips // 2), gb.tail(num_clips // 2), small_df])

    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.sort_values(by=["Speaker", "Score"])["Score"].to_csv(
        output, index=True, header=True
    )
    print(f"Wrote {len(out_df)} clips CSV to {output}")


if __name__ == "__main__":
    main()
