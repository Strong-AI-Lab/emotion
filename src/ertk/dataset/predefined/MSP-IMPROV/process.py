"""Process the raw MSP-IMPROV dataset.

This assumes the file structure from the original compressed file:
/.../
    Evaluation.txt
    session1/
        S01A/
            P/
                *.avi
                *.wav
            ...
        ...
    ...
"""

import re
from pathlib import Path

import click
import numpy as np
import pandas as pd

from ertk.dataset import resample_audio, write_annotations, write_filelist
from ertk.stats import alpha

REGEX = re.compile(
    r"^UTD-IMPROV-([A-Z0-9-]+)\.avi; ([A-Z]); A:(\d+\.\d+|NaN); V:(\d+\.\d+|NaN); D:(\d+\.\d+|NaN) ; N:(\d+\.\d+|NaN);$"  # noqa
)
RATER_RE = re.compile(
    r"^([A-Za-z0-9-_]+); (Sad|Angry|Happy|Neutral|Other); (.*); A:(\d+\.\d+|NaN); V:(\d+\.\d+|NaN); D:(\d+\.\d+|NaN); N:(\d+\.\d+|NaN);$"  # noqa
)

emotion_map = {
    "A": "anger",
    "H": "happiness",
    "S": "sadness",
    "N": "neutral",
    "O": "other",
    "X": "unknown",
}

ratings_map = {"angry": "anger", "happy": "happiness", "sad": "sadness"}

unused_emotions = ["O", "X"]


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the MSP-IMPROV dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    dimensions = {}
    labels = {}
    _ratings = []
    with open(input_dir / "Evaluation.txt") as fid:
        name = ""
        for line in fid:
            line = line.strip()
            match = REGEX.match(line)
            match_rater = RATER_RE.match(line)
            if match:
                name = "MSP-IMPROV-" + match.group(1)
                labels[name] = match.group(2)
                dimensions[name] = list(map(float, match.group(3, 4, 5, 6)))
            elif match_rater:
                rater = match_rater.group(1).strip()
                label = match_rater.group(2).strip().lower()
                label = ratings_map.get(label, label)
                # other_labels = match_rater.group(3).strip()
                act = float(match_rater.group(4).strip())
                val = float(match_rater.group(5).strip())
                dom = float(match_rater.group(6).strip())
                nat = float(match_rater.group(7).strip())
                _ratings.append((name, rater, label, act, val, dom, nat))

    paths = list(input_dir.glob("session?/*/*/*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        paths = list(resample_dir.glob("*.wav"))
    write_filelist(paths, "files_all")
    write_filelist(
        [p for p in paths if labels[p.stem] not in unused_emotions],
        "files_4class",
    )

    write_annotations({n: emotion_map[labels[n]] for n in labels}, "label")
    write_annotations({p.stem: emotion_map[p.stem[14]] for p in paths}, "acted_label")
    write_annotations({p.stem: p.stem[20] for p in paths}, "recording")
    write_annotations({p.stem: p.stem[12:14] for p in paths}, "sentence")
    speaker_dict = {p.stem: p.stem[16:19] for p in paths}
    write_annotations(speaker_dict, "speaker")
    write_annotations({k: v[0] for k, v in speaker_dict.items()}, "gender")
    write_annotations({k: v[-2:] for k, v in speaker_dict.items()}, "session")
    write_annotations({p.stem: "en" for p in paths}, "language")
    write_annotations({p.stem: "us" for p in paths}, "country")

    # Aggregated dimensional annotations per utterance
    df = pd.DataFrame.from_dict(
        dimensions,
        orient="index",
        columns=["activation", "valence", "dominance", "naturalness"],
    )
    df.index.name = "name"
    for dim in ["activation", "valence", "dominance", "naturalness"]:
        write_annotations(df[dim].to_dict(), dim)

    #
    # Ratings analysis
    #
    ratings = pd.DataFrame(
        sorted(_ratings), columns=["name", "rater", "label", "act", "val", "dom", "nat"]
    )
    print("Number of duplicated raters per clip:")
    print(ratings.groupby(["name", "rater"]).size().value_counts().loc[2:])
    # They appear to simply be copied
    ratings.drop_duplicates(["name", "rater"], inplace=True)
    ratings.set_index(["name", "rater"]).to_csv("ratings.csv")

    # Frequency of mode label(s) per utterance
    name_groups = ratings.groupby("name")
    mode_labels = (
        name_groups["label"]
        .value_counts(sort=True, ascending=False)
        .rename("freq")
        .reset_index(level="label")
        .groupby(level="name")
        .first()
        .join(name_groups.size().rename("total"))
    )

    # Include only names with a label which is strictly a plurality
    mode_labels = mode_labels.loc[
        name_groups["label"].agg(lambda x: "".join(x.mode())).isin(set("NASH"))
    ]
    print(f"{len(mode_labels)} clips have plurality vote")

    # Agreement is mean proportion of labels which are plurality label
    agreement = np.mean(mode_labels["freq"] / mode_labels["total"])
    print(f"Mean label agreement: {agreement:.3f}")

    # Acted label
    mode_labels["acted"] = [x[14] for x in mode_labels.index]

    # Agreement with acted label
    agreement = (mode_labels["label"] == mode_labels["acted"]).mean()
    print(f"Plurality vote accuracy to acted: {agreement:.3f}")

    clips = ratings.merge(mode_labels, on="name", suffixes=("_human", "_vote"))
    accuracy = (clips["label_human"] == clips["label_vote"]).mean()
    print(f"Human accuracy to plurality vote: {accuracy:.3f}")
    accuracy = (clips["label_human"] == clips["acted"]).mean()
    print(f"Human accuracy to acted: {accuracy:.3f}")

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
