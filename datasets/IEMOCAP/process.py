"""Process the raw IEMOCAP dataset.

This assumes the file structure from the original compressed file:
/.../
    Session1/
        sentences/
            wav/
                Ses01F_impro01/
                    *.wav
                ...
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

# [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
REGEX = re.compile(
    r"^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\w+)\t(\w+)\t\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]$"  # noqa
)
TRN_REGEX = re.compile(
    r"^(Ses0[1-5][MF]_(?:impro|script)0[1-9][ab]?(?:_\db?)?_[MF][X\d]{2}\d) \[\d+\.\d+-\d+\.\d+\]:(.*)$"  # noqa
)

emotion_map = {
    "ang": "anger",
    "hap": "happiness",
    "sad": "sadness",
    "neu": "neutral",
    "exc": "excitement",
    "dis": "disgust",
    "fru": "frustration",
    "fea": "fear",
    "sur": "surprise",
    "oth": "other",
    "xxx": "unknown",
}

ratings_map = {"excited": "excitement"}


def _clean(words: str):
    words = " ".join(re.split(r"\[\w+\]", words))
    words = " ".join(words.split("..."))
    words = " ".join(words.split(".."))
    words = " ".join(words.split("-"))
    return " ".join(x.strip() for x in words.split())


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the IEMOCAP dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    dimensions = {}
    labels = {}
    _cat_ratings = []
    _att_ratings = []
    for filename in input_dir.glob("Session?/dialog/EmoEvaluation/*.txt"):
        with open(filename) as fid:
            name = ""
            for line in fid:
                line = line.strip()
                match = REGEX.match(line)
                if match:
                    name = match.group(3)
                    labels[name] = match.group(4)
                    dimensions[name] = list(map(float, match.group(5, 6, 7)))
                elif line.startswith("C"):
                    # Emotion classification
                    rater, _annotations = line.strip().split(":")
                    rater = rater.strip().split("-")[1]
                    if rater[0] in "MF":
                        # M or F refer to self-evalulation
                        continue
                    *annotations, comments = _annotations.strip().split(";")
                    label = annotations[0].strip().lower()
                    _cat_ratings.append((name, rater, ratings_map.get(label, label)))
                elif line.startswith("A"):
                    # Emotion attributes
                    rater, _annotations = line.strip().split(":")
                    rater = rater.strip().split("-")[1]
                    if rater[0] in "MF":
                        # M or F refer to self-evalulation
                        continue
                    *annotations, comments = _annotations.strip().split(";")
                    val = int(annotations[0].strip().split(" ")[1])
                    act = int(annotations[1].strip().split(" ")[1])
                    try:
                        # For some reason 36 of these are missing
                        dom = int(annotations[2].strip().split("  ")[1])
                    except IndexError:
                        dom = 0
                    _att_ratings.append((name, rater, val, act, dom))

    paths = list(input_dir.glob("Session?/sentences/wav/*/*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        paths = list(resample_dir.glob("*.wav"))
    write_filelist(paths, "files_all")
    write_filelist(
        [p for p in paths if labels[p.stem] in ["ang", "hap", "neu", "sad", "exc"]],
        "files_4class",
    )

    write_annotations({n: emotion_map[labels[n]] for n in labels}, "label")
    speaker_dict = {p.stem: p.stem[3:6] for p in paths}
    write_annotations(speaker_dict, "speaker")
    write_annotations({k: v[-1] for k, v in speaker_dict.items()}, "gender")
    write_annotations({k: v[:2] for k, v in speaker_dict.items()}, "session")
    write_annotations(
        {p.stem: "impro" if "impro" in p.stem else "script" for p in paths}, "sess_type"
    )
    write_annotations({p.stem: "en" for p in paths}, "language")
    write_annotations({p.stem: "us" for p in paths}, "country")

    # Aggregated dimensional annotations per utterance
    df = pd.DataFrame.from_dict(
        dimensions, orient="index", columns=["valence", "activation", "dominance"]
    )
    df.index.name = "name"
    for dim in ["valence", "activation", "dominance"]:
        df[dim].to_csv(dim + ".csv", index=True, header=True)

    dim_ratings = (
        pd.DataFrame(_att_ratings, columns=["name", "rater", "val", "act", "dom"])
        .set_index(["name", "rater"])
        .sort_index()
    )
    cat_ratings = (
        pd.DataFrame(_cat_ratings, columns=["name", "rater", "label"])
        .set_index(["name", "rater"])
        .sort_index()
    )
    ratings = cat_ratings.join(dim_ratings)
    ratings.to_csv("ratings.csv")

    ratings["label"] = ratings["label"].str.lower()
    ratings["label"] = ratings["label"].map(
        lambda x: {"excited": "excitement"}.get(x, x)
    )
    ratings["label"] = ratings["label"].astype("category")

    # There are 3 (non-self) raters per utterance. Agreement is the
    # proportion of labels which are the same. This formula only works
    # for 3 raters.
    mode_count = 4 - ratings["label"].unstack().nunique(1)
    agreement = np.mean(mode_count / 3)
    print(mode_count.value_counts().rename_axis("majority").to_frame("count"))
    print(f"Mean label agreement: {agreement:.3f}")

    # Simple way to get int matrix of labels for raters x clips
    data = ratings["label"].cat.codes.unstack() + 1
    data[data.isna()] = 0
    data = data.astype(int).to_numpy()
    print(f"Krippendorf's alpha (categorical): {alpha(data):.3f}")

    # Transcriptions
    utts = {}
    for p in input_dir.glob("Session?/dialog/transcriptions/*.txt"):
        with open(p) as fid:
            for line in map(str.strip, fid):
                match = TRN_REGEX.match(line)
                if match:
                    utts[match.group(1)] = match.group(2).strip()
    utts = {n: _clean(u) for n, u in utts.items()}
    df = pd.DataFrame.from_dict(utts, orient="index", columns=["transcript"])
    df = df.rename_axis(index="name").sort_index()
    df.to_csv("transcript.csv", header=True, index=True)


if __name__ == "__main__":
    main()
