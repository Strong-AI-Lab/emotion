"""Process the raw ESD dataset.

This assumes the file structure from the original compressed file:
/.../
    ReadMe.txt
    0001/
        0001.txt
        Angry/
            train/
            evaluation/
            test/
        Happy/
            train/
            evaluation/
            test/
        ...
    0002/
    ...
"""

import shutil
import unicodedata
from pathlib import Path

import click
from joblib import delayed

from ertk.dataset import write_annotations, write_filelist
from ertk.utils import TqdmParallel

emotion_map = {
    "Angry": "anger",
    "Happy": "happiness",
    "Neutral": "neutral",
    "Sad": "sadness",
    "Surprise": "surprise",
}

encodings = {
    1: "gb2312",
    2: "gb2312",
    3: "utf_16",
    4: "gb2312",
    5: "gb2312",
    6: "utf_16",
    7: "utf_16",
    8: "utf_16",
    9: "gb2312",
    10: "utf_16",
    11: "utf_8",
    12: "utf_16",
    13: "utf_16",
    14: "utf_16",
    15: "utf_8",
    16: "latin_1",
    17: "latin_1",
    18: "utf_16",
    19: "utf_16",
    20: "utf_8",
}

error_lines = {
    "0012_001355": "The fisherman and his wife see George every day.",
    "0013_000431": "And I never had a whooping cough why.",
    "0013_000914": "But what about this thing, sticky!",
    "0015_000137": "Let the glass globe be.",
    "0015_000138": "But one requires the explorer to furnish proofs.",
    "0015_000175": "All this have we won by our labour.",
    "0015_000187": "An hour out of Guildford town.",
    "0015_000217": "You cruelty shall cost your life !",
    "0015_000668": "The name of the song is called haddocks.",
    "0015_000690": "So Tom saw night as it were broad daylight.",
    "0015_000875": "All this have we won by our labour.",
    "0015_001015": "It's to say how do you do with Tom's answer.",
    "0015_001531": "It must come sometimes to jam a day.",
    "0015_001578": "And vowed he'd change the pigtail's place.",
    "0014_001590": "And they were sandy yellow brownish all over.",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the ESD dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("**/*.wav"))
    subsets = {}
    for subset in ["train", "evaluation", "test"]:
        subsets[subset] = {x.name for x in paths if subset in x.parts}

    resample_dir = Path("resampled")
    if resample:
        resample_dir.mkdir(exist_ok=True)
        TqdmParallel(len(paths), "Copying files")(
            delayed(shutil.copyfile)(x, resample_dir / x.name) for x in paths
        )
        paths = list(resample_dir.glob("*.wav"))
    write_filelist(paths, "files_all")
    for subset in ["train", "evaluation", "test"]:
        write_filelist(subsets[subset], f"files_{subset}")

    transcripts = {}
    for spk_id in range(1, 21):
        trans_file = input_dir / f"00{spk_id:02d}" / f"00{spk_id:02d}.txt"
        with open(trans_file, encoding=encodings[spk_id]) as fid:
            for line in fid:
                line = line.strip()
                if line:
                    try:
                        name, transcript, _ = line.split("\t")
                    except ValueError:
                        continue
                    transcripts[name] = unicodedata.normalize("NFKD", transcript)
    transcripts.update(error_lines)
    write_annotations(transcripts, "transcript")
    write_annotations(
        {k: v for k, v in transcripts.items() if int(k[:4]) < 11}, "transcripts_zh"
    )
    write_annotations(
        {k: v for k, v in transcripts.items() if int(k[:4]) >= 11}, "transcripts_en"
    )

    labels = {}
    for emo1, emo2 in emotion_map.items():
        labels.update({x.stem: emo2 for x in input_dir.glob(f"*/{emo1}/*/*.wav")})
    write_annotations(labels, "label")

    speaker_dict = {p.stem: p.stem[:4] for p in paths}
    write_annotations(speaker_dict, "speaker")
    male_speakers = {4, 5, 6, 8, 10, 11, 12, 13, 14, 20}
    write_annotations(
        {k: "M" if int(v) in male_speakers else "F" for k, v in speaker_dict.items()},
        "gender",
    )
    write_annotations(
        {k: "zh" if int(v) < 11 else "en" for k, v in speaker_dict.items()}, "language"
    )


if __name__ == "__main__":
    main()
