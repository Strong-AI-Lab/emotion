"""Process the raw CaFE dataset.

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

from ertk.dataset import write_annotations, write_filelist
from ertk.stats import alpha
from ertk.utils import PathlibPath

emotion_map = {
    "A": "anger",
    "D": "disgust",
    "F": "fear",
    "H": "happiness",
    "S": "sadness",
    "N": "neutral",
    # For multiple modes, as in MSP-IMPROV
    "X": "unknown"
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process CREMA-D dataset at location INPUT_DIR."""

    paths = list(input_dir.glob("AudioWAV/*.wav"))
    # 1076_MTI_SAD_XX has no audio signal
    write_filelist([p for p in paths if p.stem != "1076_MTI_SAD_XX"])
    write_annotations({p.stem: emotion_map[p.stem[9]] for p in paths}, "label")
    write_annotations({p.stem: p.stem[:4] for p in paths}, "speaker")
    write_annotations({p.stem: "en" for p in paths}, "language")

    summaryTable = pd.read_csv(
        input_dir / "processedResults" / "summaryTable.csv",
        low_memory=False,
        index_col=0,
    )
    summaryTable["ActedEmo"] = summaryTable["FileName"].apply(lambda x: x[9])

    for mode in ["VoiceVote", "FaceVote", "MultiModalVote"]:
        # Proportion of majority vote equivalent to acted emotion
        accuracy = (summaryTable[mode] == summaryTable["ActedEmo"]).sum() / len(
            summaryTable
        )
        print(f"Acted accuracy using {mode}: {accuracy:.3f}")
    print()

    # Majority vote annotations from other modalities
    valid = summaryTable["MultiModalVote"].isin(list("NHDFAS"))
    summaryTable.loc[~valid, "MultiModalVote"] = "X"
    labels = dict(zip(summaryTable["FileName"], summaryTable["MultiModalVote"]))
    write_annotations(labels, "label_multimodal")

    valid = summaryTable["FaceVote"].isin(list("NHDFAS"))
    summaryTable.loc[~valid, "FaceVote"] = "X"
    labels = dict(zip(summaryTable["FileName"], summaryTable["FaceVote"]))
    write_annotations(labels, "label_face")

    valid = summaryTable["VoiceVote"].isin(list("NHDFAS"))
    summaryTable.loc[~valid, "VoiceVote"] = "X"
    labels = dict(zip(summaryTable["FileName"], summaryTable["VoiceVote"]))
    write_annotations(labels, "label_voice")

    finishedResponses = pd.read_csv(
        input_dir / "finishedResponses.csv", low_memory=False, index_col=0
    )
    finishedResponses["respLevel"] = pd.to_numeric(
        finishedResponses["respLevel"], errors="coerce"
    )
    # Remove these two duplicates
    finishedResponses = finishedResponses.drop([137526, 312184], errors="ignore")

    finishedEmoResponses = pd.read_csv(
        input_dir / "finishedEmoResponses.csv", low_memory=False, index_col=0
    )
    finishedEmoResponses = finishedEmoResponses.query(
        "clipNum != 7443 and clipNum != 7444"
    )
    distractedResponses = finishedEmoResponses.query("ttr > 10000")

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
    # Get all annotations not defined to be distracted
    goodResponses = finishedResponses[~uniqueIDs.isin(distractedIDs)]

    # Responses based on different modalities
    voiceResp = goodResponses.query("queryType == 1")
    faceResp = goodResponses.query("queryType == 2")
    multiModalResp = goodResponses.query("queryType == 3")

    resp_d = {"voice": voiceResp, "face": faceResp, "both": multiModalResp}
    for s, df in resp_d.items():
        # Proportion of human responses equal to acted emotion
        accuracy = (df["respEmo"] == df["dispEmo"]).sum() / len(df)
        print(f"Human accuracy using {s}: {accuracy:.3f}")

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
    tabulatedVotes["mode"] = tabulatedVotes.index // 100000
    tabulatedVotes["mode"] = tabulatedVotes["mode"].map(
        lambda x: [None, "voice", "face", "both"][x]
    )
    print("Average vote agreement per annotation mode:")
    print(tabulatedVotes.groupby("mode")["agreement"].describe())


if __name__ == "__main__":
    main()
