"""Process the raw MSP-PODCAST dataset.

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
from pathlib import Path

import click
import numpy as np
import pandas as pd

from ertk.dataset import write_annotations, write_filelist
from ertk.stats import alpha

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

rating_map = {"angry": "anger", "happy": "happiness", "sad": "sadness"}


def _filter_other(s: str) -> str:
    if s.startswith("Other"):
        s = "Other"
    s = s.lower()
    return rating_map.get(s, s)


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--resample/--noresample", default=False, help="Resample audio to local directory."
)
def main(input_dir: Path, resample: bool):
    """Process the MSP-IMPROV dataset at location INPUT_DIR."""

    labels_dir = input_dir / "Labels"
    labels_concensus = pd.read_csv(
        labels_dir / "labels_concensus.csv",
        header=0,
        low_memory=False,
        na_values=["Unknown"],
        dtype={"SpkrID": str},
    )
    labels_concensus.set_index("FileName", inplace=True)
    labels_concensus.index = labels_concensus.index.map(lambda x: x[:-4])
    for dim in ["EmoAct", "EmoVal", "EmoDom"]:
        write_annotations(labels_concensus[dim].to_dict(), dim)
    labels = labels_concensus["EmoClass"].to_dict()

    audios_dir = input_dir / "Audios"
    paths = list(audios_dir.glob("*.wav"))
    write_filelist(paths, "files_all")
    split_sets = labels_concensus.groupby("Split_Set").groups
    for fileset, names in split_sets.items():
        write_filelist(
            names.map(lambda x: audios_dir / (x + ".wav")), f"files_{fileset}"
        )

    write_annotations({n: emotion_map[labels[n]] for n in labels}, "label")
    write_annotations(labels_concensus["Gender"].to_dict(), "gender")
    write_annotations(labels_concensus["SpkrID"].to_dict(), "speaker")
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
                _ratings.append(
                    (
                        name[:-4],
                        worker,
                        _filter_other(annotations["EmoClass_Major"]),
                        annotations["EmoAct"],
                        annotations["EmoDom"],
                        annotations["EmoVal"],
                    )
                )
    ratings = pd.DataFrame(
        sorted(_ratings), columns=["name", "rater", "label", "act", "val", "dom"]
    )
    ratings.set_index(["name", "rater"]).to_csv("ratings.csv")

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
