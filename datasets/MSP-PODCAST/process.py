"""Process the raw MSP-IMPROV dataset.

This assumes the file structure from the original compressed file:
/.../
    Labels/
        labels_concensus.csv
        ...
    Audios/
        *.wav
    readme.txt
    ...
"""


import json
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd

from ertk.dataset import write_annotations, write_filelist
from ertk.stats import alpha
from ertk.utils import PathlibPath

REGEX = re.compile(
    r"^UTD-IMPROV-([A-Z0-9-]+)\.avi; ([A-Z]); A:(\d+\.\d+|NaN); V:(\d+\.\d+|NaN); D:(\d+\.\d+|NaN) ; N:(\d+\.\d+|NaN);$"  # noqa
)

emotion_map = {
    "A": "anger",
    "H": "happiness",
    "S": "sadness",
    "N": "neutral",
    "U": "surprise",
    "F": "fear",
    "D": "disgust",
    "C": "contempt",
    "O": "other",
    "X": "unknown",
}

unused_emotions = ["O", "X"]


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the MSP-IMPROV dataset at location INPUT_DIR."""

    labels_dir = input_dir / "Labels"
    labels_concensus = pd.read_csv(labels_dir / "labels_concensus.csv", header=0)
    labels_concensus.set_index("FileName", inplace=True)
    labels_concensus.index = labels_concensus.index.map(lambda x: x[:-4])
    for dim in ["EmoAct", "EmoVal", "EmoDom"]:
        write_annotations(labels_concensus[dim].to_dict(), dim)
    labels = labels_concensus["EmoClass"].to_dict()

    audios_dir = input_dir / "Audios"
    paths = list(audios_dir.glob("*.wav"))
    write_filelist(paths, "files_all.txt")
    split_sets = labels_concensus.groupby("Split_Set").groups
    for fileset, names in split_sets.items():
        write_filelist(
            names.map(lambda x: audios_dir / (x + ".wav")),
            f"files_{fileset}.txt",
        )

    write_annotations({n: emotion_map[labels[n]] for n in labels}, "label")
    unk_spk = labels_concensus["SpkrID"] == "Unknown"  # Ignore unknown speakers
    write_annotations(labels_concensus.loc[unk_spk, "Gender"].to_dict(), "gender")
    write_annotations(labels_concensus.loc[unk_spk, "SpkrID"].to_dict(), "speaker")
    write_annotations(labels_concensus["Split_Set"].to_dict(), "split")
    write_annotations({p.stem: "en" for p in paths}, "language")

    #
    # Ratings analysis
    #
    with open(labels_dir / "labels_detailed.json") as fid:
        labels_detailed = json.load(fid)
        _ratings = []
        for name, workers in labels_detailed.items():
            for worker, annotations in workers.items():
                _ratings.append((name[:-4], worker, annotations["EmoClass_Major"][0]))
    ratings = pd.DataFrame(sorted(_ratings), columns=["name", "rater", "label"])
    ratings.drop_duplicates(["name", "rater"], inplace=True)

    num_ratings = ratings.groupby("name").size().to_frame("total")
    label_count = ratings.groupby(["name", "label"]).size().to_frame("freq")
    # Count of majority label per utterance
    mode_count = (
        label_count.reset_index()
        .sort_values("freq", ascending=False)
        .drop_duplicates(subset="name")
        .set_index("name")
        .join(num_ratings)
        .sort_index()
    )

    # Include only names with a label which is strictly a plurality
    mode_count = mode_count[
        ratings.groupby("name")["label"]
        .agg(lambda x: "".join(x.mode()))
        .map(lambda x: len(x) == 1)
        .sort_index()
    ]

    # Agreement is mean proportion of labels which are plurality label
    agreement = np.mean(mode_count["freq"] / mode_count["total"])
    print(f"Mean label agreement: {agreement:.3f}")

    clips = ratings.join(mode_count["label"], "name", rsuffix="_vote", how="inner")
    accuracy = (clips["label"] == clips["label_vote"]).sum() / len(clips)
    print(f"Human accuracy: {accuracy:.3f}")

    # Simple way to get int matrix of labels for raters x clips
    data = (
        ratings.set_index(["rater", "name"])["label"]
        .astype("category")
        .cat.codes.unstack()
        + 1
    )
    data[data.isna()] = 0
    data = data.astype(int).to_numpy()
    print(f"Krippendorf's alpha: {alpha(data):.3f}")


if __name__ == "__main__":
    main()
