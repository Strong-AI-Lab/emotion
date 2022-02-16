"""Process the raw ShEMO dataset.

This assumes the file structure from the original compressed file:
/.../
    male/
        *.wav
    female/
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist

emotion_map = {
    "A": "anger",
    "H": "happiness",
    "N": "neutral",
    "S": "sadness",
    "W": "surprise",
    "F": "fear",
}

unused_emotions = ["F"]


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the ShEMO dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("*/*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all")
        write_filelist(
            [p for p in resample_dir.glob("*.wav") if p.stem[3] not in unused_emotions],
            "files_5class",
        )

    write_annotations({p.stem: emotion_map[p.stem[3]] for p in paths}, "label")
    speaker_dict = {p.stem: p.stem[:3] for p in paths}
    write_annotations(speaker_dict, "speaker")
    write_annotations({k: v[0] for k, v in speaker_dict.items()}, "gender")
    write_annotations({p.stem: "ar" for p in paths}, "language")
    write_annotations({p.stem: "ir" for p in paths}, "country")


if __name__ == "__main__":
    main()
