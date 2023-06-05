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
from tqdm import tqdm

from ertk.dataset import write_annotations, write_filelist

sentence_map = {
    "a01": "Der Lappen liegt auf dem Eisschrank.",
    "a02": "Das will sie am Mittwoch abgeben.",
    "a04": "Heute abend könnte ich es ihm sagen.",
    "a05": "Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.",
    "a07": "In sieben Stunden wird es soweit sein.",
    "b01": "Was sind denn das für Tüten, die da unter dem Tisch stehen?",
    "b02": "Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.",
    "b03": "An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.",  # noqa E501
    "b09": "Ich will das eben wegbringen und dann mit Karl was trinken gehen.",
    "b10": "Die wird auf dem Platz sein, wo wir sie immer hinlegen.",
}

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

    speaker_dict = {p.stem: p.stem[:2] for p in paths}
    write_annotations(speaker_dict, "speaker")
    male_speakers = ["03", "10", "11", "12", "15"]
    gender_dict = {
        k: "M" if v in male_speakers else "F" for k, v in speaker_dict.items()
    }
    write_annotations(gender_dict, "gender")
    write_annotations({p.stem: p.stem[2:5] for p in paths}, "sentence")
    write_annotations({p.stem: sentence_map[p.stem[2:5]] for p in paths}, "transcript")
    write_annotations({p.stem: emotion_map[p.stem[5]] for p in paths}, "label")
    write_annotations({p.stem: p.stem[6] for p in paths}, "version")
    write_annotations({p.stem: "de" for p in paths}, "language")

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
    write_annotations(utt, "syllables")

    if resample:
        resample_dir = Path("resampled")
        resample_dir.mkdir(exist_ok=True)
        for p in tqdm(paths, desc="Copying audio"):
            shutil.copyfile(p, resample_dir / p.name)
        paths = list(resample_dir.glob("*.wav"))
    write_filelist(paths, "files_all")


if __name__ == "__main__":
    main()
