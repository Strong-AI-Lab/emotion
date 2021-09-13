"""Process the raw CaFE dataset.

This assumes the file structure from the original compressed file:
/.../
    AudioMP3/
        *.mp3 [for 44.1 kHz]
    AudioWAV/
        *.wav [for 16 kHz]
    ...
"""

import shutil
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

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
    "X": "unknown",
}


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process CREMA-D dataset at location INPUT_DIR."""

    paths = list(input_dir.glob("AudioWAV/*.wav"))
    write_annotations({p.stem: emotion_map[p.stem[9]] for p in paths}, "label")
    write_annotations({p.stem: p.stem[:4] for p in paths}, "speaker")
    write_annotations({p.stem: "en" for p in paths}, "language")
    write_annotations({p.stem: "us" for p in paths}, "country")
    # 1076_MTI_SAD_XX has no signal, 1040_ITH_SAD_X is incorrectly named
    paths = [p for p in paths if p.stem not in {"1076_MTI_SAD_XX", "1040_ITH_SAD_X"}]
    if resample:
        Path("resampled").mkdir(exist_ok=True)
        for p in tqdm(paths, desc="Copying audio"):
            shutil.copyfile(p, Path("resampled", p.name))
    write_filelist(Path("resampled").glob("*.wav"), "files_all")

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
    finishedEmoResponses = finishedEmoResponses[
        ~finishedEmoResponses["clipNum"].isin([7443, 7444])
    ]
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
    # Get all annotations not defined to be distracted
    goodResponses = finishedResponses[~uniqueIDs.isin(distractedIDs)]

    # Responses based on different modalities
    voiceResp = goodResponses.query("queryType == 1")
    faceResp = goodResponses.query("queryType == 2")
    multiModalResp = goodResponses.query("queryType == 3")

    resp_d = {"voice": voiceResp, "face": faceResp, "both": multiModalResp}
    for s, df in resp_d.items():
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
