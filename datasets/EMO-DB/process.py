"""Process the raw EMO-DB dataset.

This assumes the file structure from the original compressed file:
/.../
    wav_corpus/
        *.wav
    silb/
        ...
    ...
"""

import shutil
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from ertk.dataset import write_annotations, write_filelist

emotion_map = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the EMO-DB dataset at location INPUT_DIR."""
    paths = list(input_dir.glob("wav_corpus/*.wav"))
    write_annotations({p.stem: emotion_map[p.stem[5]] for p in paths}, "label")
    speaker_dict = {p.stem: p.stem[:2] for p in paths}
    write_annotations(speaker_dict, "speaker")
    male_speakers = ["03", "10", "11", "12", "15"]
    gender_dict = {
        k: "M" if v in male_speakers else "F" for k, v in speaker_dict.items()
    }
    write_annotations(gender_dict, "gender")
    write_annotations({p.stem: "de" for p in paths}, "language")
    resample_dir = Path("resampled")
    if resample:
        resample_dir.mkdir(exist_ok=True)
        for p in tqdm(paths, desc="Copying audio"):
            shutil.copyfile(p, resample_dir / p.name)
    write_filelist(resample_dir.glob("*.wav"), "files_all")

    utt = {}
    for p in input_dir.glob("silb/*.silb"):
        with open(p, encoding="latin_1") as fid:
            words = []
            for line in fid:
                line = line.strip()
                _, word = line.split()
                if word in [".", "("]:
                    continue
                words.append(word.strip())
            utt[p.stem] = " ".join(words)

    df = pd.DataFrame({"Name": utt.keys(), "Transcript": utt.values()})
    df.sort_values("Name").to_csv("transcript.csv", index=False, header=True)


if __name__ == "__main__":
    main()
