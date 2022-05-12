"""Process the raw VENEC dataset.

This assumes the file structure from the original compressed file:
/.../
    NoNameProsody/
        *.mp3
        *.mp3.wav [duplicated by mistake?]
    CowenEtAlAnalysisCode/
        ...
    veneccountryinfo.csv
    ...
"""

import re
from pathlib import Path

import click
import pandas as pd

from ertk.dataset import resample_audio, write_annotations, write_filelist

MED_INT_REGEX = re.compile(r"^(\d\d?)([A-Za-z]+)$")
HIGH_INT_REGEX = re.compile(r"^[A-Za-z]+(\d+)([A-Za-z]+)\d?(?:[Hh]igh|Hi|H)$")
EMOTIONS = {
    "aff": "affection",
    "amu": "amusement",
    "ang": "anger",
    "con": "contempt",
    "disg": "disgust",
    "dist": "distress",
    "fea": "fear",
    "gui": "guilt",
    "hap": "happiness",
    "int": "interest",
    "lus": "lust",
    "negs": "negativesurprise",
    "negsurprise": "negativesurprise",
    "neu": "neutral",
    "nsur": "negativesurprise",
    "nsu": "negativesurprise",
    "poss": "positivesurprise",
    "possurprise": "positivesurprise",
    "psur": "positivesurprise",
    "pri": "pride",
    "rel": "relief",
    "sad": "sadness",
    "ser": "serenity",
    "sha": "shame",
    "sham": "shame",
}


def get_speaker(s: str):
    try:
        return MED_INT_REGEX.match(s)[1]
    except TypeError:
        return HIGH_INT_REGEX.match(s)[1]


def get_emotion(s: str):
    try:
        emo = MED_INT_REGEX.match(s)[2].lower()
    except TypeError:
        emo = HIGH_INT_REGEX.match(s)[2].lower()
    return EMOTIONS.get(emo, emo)


NO_SIGNAL = {"1528", "0191", "2080", "0723"}
MISSING_LABEL = {"0731", "1370", "2370", "2505"}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the VENEC dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = [
        x
        for x in input_dir.glob("NoNameProsody/*.mp3")
        if x.stem not in NO_SIGNAL | MISSING_LABEL
    ]
    resample_dir = Path("resampled")
    if resample:
        resample_audio(paths, resample_dir)
    write_filelist(resample_dir.glob("*.wav"), "files_all")

    clip_info = pd.read_csv(
        input_dir / "veneccountryinfo.csv",
        index_col=0,
        names=["Filename", "Country", "Original"],
    )
    clip_info.index = clip_info.index.map(lambda x: x[:-4])

    # Acted labels
    write_annotations(
        clip_info["Original"].map(lambda x: Path(x).stem).map(get_emotion).to_dict(),
        "label",
    )
    write_annotations(
        clip_info["Original"].map(lambda x: Path(x).parts[0].split("_")[0]).to_dict(),
        "intensity",
    )

    csvfile = input_dir / "CategoryRatingsUSA.csv"
    ratings = pd.read_csv(csvfile, index_col=0, header=0)
    ratings.index = ratings.index.map(lambda x: x[:-4])

    mode_count = ratings.apply(lambda x: x.value_counts().sort_index().iloc[-1], axis=1)
    ratings["Freq"] = ratings.max(1)
    ratings["Majority"] = ratings.columns[ratings.to_numpy().argmax(1)]
    ratings["Num_Majority"] = mode_count == 1

    agreement = ratings["Freq"].mean()
    print(f"Mean label agreement: {agreement:.3f}")

    ratings = ratings.join(clip_info)
    ratings["Country"] = ratings["Country"].map(str.upper)
    # Correct the three errors where country is STR
    ratings.loc[ratings["Country"] == "STR", "Country"] = "AUS"
    # Intensity is High or Medium
    ratings["Intensity"] = ratings["Original"].map(
        lambda x: Path(x).parts[0].split("_")[0]
    )
    ratings["Speaker_ID"] = (
        ratings["Country"]
        + "_"
        + ratings["Original"]
        .map(lambda x: Path(x).stem)
        .map(get_speaker)
        .map(int)
        .map("{:02d}".format)
    )
    ratings["Emotion"] = (
        ratings["Original"].map(lambda x: Path(x).stem).map(get_emotion)
    )

    ratings_18class = ratings[ratings["Emotion"] != "neutral"]

    # (country, speaker) pairs for which there are clips for all
    # emotions. This gives around 2262 clips.
    ratings_complete = ratings_18class.groupby("Speaker_ID").filter(
        lambda x: set(x["Emotion"]) >= set(EMOTIONS.values()) - {"neutral"}
    )

    write_filelist(
        [p for p in resample_dir.glob("*.wav") if p.stem in ratings_18class.index],
        "files_18class",
    )
    write_filelist(
        [p for p in resample_dir.glob("*.wav") if p.stem in ratings_complete.index],
        "files_18class_complete",
    )
    speaker_dict = ratings["Speaker_ID"].to_dict()
    write_annotations(speaker_dict, "speaker")
    write_annotations({k: v[:3] for k, v in speaker_dict.items()}, "country")
    write_annotations({p.stem: "en" for p in paths}, "language")


if __name__ == "__main__":
    main()
