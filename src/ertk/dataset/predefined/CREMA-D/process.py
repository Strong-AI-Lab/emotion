"""Process the raw CREMA-D dataset.

This assumes the file structure from the original compressed file:
/.../
    AudioMP3/
        *.mp3 [for 44.1 kHz]
    AudioWAV/
        *.wav [for 16 kHz]
    ...
"""

from pathlib import Path

import click
import pandas as pd

from ertk.dataset import resample_audio, write_annotations, write_filelist
from ertk.stats import alpha

emotion_map = {
    "A": "anger",
    "D": "disgust",
    "F": "fear",
    "H": "happiness",
    "S": "sadness",
    "N": "neutral",
    # For multiple modes, as in MSP-IMPROV
    "X": "unknown",
}

sentence_map = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process CREMA-D dataset at location INPUT_DIR."""

    paths = list(input_dir.glob("AudioMP3/*.mp3"))  # MP3 has fewer issues
    write_annotations({p.stem: emotion_map[p.stem[9]] for p in paths}, "label")
    speakers = {p.stem: p.stem[:4] for p in paths}
    write_annotations(speakers, "speaker")
    spk_df = pd.read_csv(
        input_dir / "VideoDemographics.csv", header=0, dtype={0: str}
    ).set_index("ActorID")
    for key in ["Age", "Sex", "Race", "Ethnicity"]:
        write_annotations(
            {k: spk_df.loc[v, key] for k, v in speakers.items()}, key.lower()
        )
    write_annotations({p.stem: p.stem[5:8] for p in paths}, "sentence")
    write_annotations({p.stem: sentence_map[p.stem[5:8]] for p in paths}, "transcript")
    write_annotations({p.stem: p.stem[-2:] for p in paths}, "level")
    write_annotations({p.stem: "en" for p in paths}, "language")
    write_annotations({p.stem: "us" for p in paths}, "country")

    if resample:
        resample_dir = Path("resampled")
        resample_dir.mkdir(exist_ok=True)
        resample_audio(paths, resample_dir)
        paths = list(resample_dir.glob("*.wav"))
    # 1076_MTI_SAD_XX has no signal
    paths = [p for p in paths if p.stem != "1076_MTI_SAD_XX"]
    write_filelist(paths, "files_all")

    # Annotations
    summaryTable = pd.read_csv(
        input_dir / "processedResults" / "summaryTable.csv",
        low_memory=False,
        index_col=1,
    )
    summaryTable["ActedEmo"] = summaryTable.index.map(lambda x: x[9])

    for mode in ["VoiceVote", "FaceVote", "MultiModalVote"]:
        # Proportion of majority vote equivalent to acted emotion
        accuracy = (summaryTable[mode] == summaryTable["ActedEmo"]).mean()
        print(f"Acted accuracy using {mode}: {accuracy:.3f}")
    print()

    # Majority vote annotations from other modalities
    valid = summaryTable["MultiModalVote"].isin(list("NHDFAS"))
    summaryTable.loc[~valid, "MultiModalVote"] = "X"
    write_annotations(summaryTable["MultiModalVote"].to_dict(), "label_multimodal")

    valid = summaryTable["FaceVote"].isin(list("NHDFAS"))
    summaryTable.loc[~valid, "FaceVote"] = "X"
    write_annotations(summaryTable["FaceVote"].to_dict(), "label_face")

    valid = summaryTable["VoiceVote"].isin(list("NHDFAS"))
    summaryTable.loc[~valid, "VoiceVote"] = "X"
    write_annotations(summaryTable["VoiceVote"].to_dict(), "label_voice")

    # Full ratings
    finishedResponses = pd.read_csv(
        input_dir / "finishedResponses.csv", low_memory=False, index_col=0
    )
    finishedResponses["respLevel"] = pd.to_numeric(
        finishedResponses["respLevel"], errors="coerce"
    )
    # Remove these two duplicates
    finishedResponses = finishedResponses.drop(index=[137526, 312184], errors="ignore")

    finishedEmoResponses = pd.read_csv(
        input_dir / "finishedEmoResponses.csv", low_memory=False, index_col=0
    )
    # Ignore practice clips
    finishedEmoResponses = finishedEmoResponses[
        ~finishedEmoResponses["clipNum"].isin([7443, 7444])
    ]

    # Get all annotations not defined to be distracted
    # Same as from processFinishedResponses.r
    distractedResponses = finishedEmoResponses[finishedEmoResponses["ttr"] > 10000]
    uniqueIDs = (
        finishedResponses["sessionNums"] * 1000
        + finishedResponses["queryType"] * 100
        + finishedResponses["questNum"]
    )
    distractedIDs = (
        distractedResponses["sessionNums"] * 1000
        + distractedResponses["queryType"] * 100
        + distractedResponses["questNum"]
    )
    goodResponses = finishedResponses[~uniqueIDs.isin(distractedIDs)]

    # Responses based on different modalities
    resp_d = {1: "voice", 2: "face", 3: "both"}
    gb = goodResponses.groupby("queryType")
    for num, s in resp_d.items():
        df = gb.get_group(num)

        ratings = (
            df.rename(columns={"clipName": "name", "localid": "rater"})
            .set_index(["name", "rater"])
            .sort_index()[["respEmo", "respLevel", "numTries", "ttr"]]
        )
        # Make sure all names are present in ratings
        assert len(ratings.index.unique("name")) == 7442
        ratings.to_csv(f"ratings_{s}.csv")

        # Proportion of human responses equal to acted emotion
        accuracy = (df["respEmo"] == df["dispEmo"]).mean()
        print(f"Human accuracy to acted using {s}: {accuracy:.3f}")

        dataTable = (
            df.set_index(["sessionNums", "clipNum"])["respEmo"]
            .astype("category")
            .cat.codes.unstack()
            + 1
        )
        dataTable[dataTable.isna()] = 0
        data = dataTable.astype(int).to_numpy()
        print(f"Krippendorf's alpha using {s}: {alpha(data):.3f}")
        print()

    tabulatedVotes = pd.read_csv(
        input_dir / "processedResults" / "tabulatedVotes.csv",
        low_memory=False,
        index_col=0,
    )
    tabulatedVotes["mode"] = tabulatedVotes.index.map(
        lambda x: ["voice", "face", "both"][x // 100000 - 1]
    )
    print("Average vote agreement per annotation mode:")
    print(tabulatedVotes.groupby("mode")["agreement"].describe())


if __name__ == "__main__":
    main()
