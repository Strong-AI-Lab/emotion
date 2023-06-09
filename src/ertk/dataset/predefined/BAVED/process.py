"""Process the raw BAVED dataset.

This assumes the file structure from the original data:
/.../
    0/
        *.wav
    1/
        *.wav
    ...
    speakers_info.json
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist

emotion_level = ["low", "normal", "high"]
word_map = [
    "اعجبني",
    "لم يعجبني",
    "هذا",
    "الفيلم",
    "رائع",
    "مقول",
    "سيئ",
]


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the BAVED dataset at location INPUT_DIR."""
    paths = list(input_dir.glob("?/*.wav"))
    resample_dir = Path("resampled")
    if resample:
        resample_audio(paths, resample_dir)
    write_filelist(resample_dir.glob("*.wav"), "files_all")

    levels = {}
    speakers = {}
    genders = {}
    ages = {}
    words = {}
    for path in paths:
        spk, gen, age, word, level, _ = path.stem.split("-")
        levels[path.stem] = emotion_level[int(level)]
        speakers[path.stem] = spk
        genders[path.stem] = gen.upper()
        ages[path.stem] = int(age)
        words[path.stem] = word_map[int(word)]

    write_annotations(levels, "level")
    write_annotations(speakers, "speaker")
    write_annotations(genders, "gender")
    write_annotations(ages, "age")
    write_annotations(words, "word")
    write_annotations({p.stem: "ar" for p in paths}, "language")


if __name__ == "__main__":
    main()
