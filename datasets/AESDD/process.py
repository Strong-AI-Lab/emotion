"""Process the raw AESDD dataset.

This assumes the file structure from the original compressed file:
/.../
    anger/
        *.wav
    disgust/
        *.wav
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist

emotion_map = {
    "a": "anger",
    "d": "disgust",
    "h": "happiness",
    "f": "fear",
    "s": "sadness",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the CaFE dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("**/*.wav"))
    names = [x.stem for x in paths]
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all")
    else:
        write_filelist(paths, "files_all")

    write_annotations({x: emotion_map[x[0]] for x in names}, "label")
    speaker_dict = {x: str(int(x[x.find("(") + 1 : x.find(")")])) for x in names}
    write_annotations(speaker_dict, "speaker")
    write_annotations(
        {k: ["F", "M"][int(v) % 2] for k, v in speaker_dict.items()},
        "gender",
    )
    write_annotations({x: "el" for x in names}, "language")


if __name__ == "__main__":
    main()
