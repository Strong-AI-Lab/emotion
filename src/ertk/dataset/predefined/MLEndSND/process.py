"""Process the raw MLEnd Spoken Numerals dataset.

This assumes the file structure from the original dataset:
/.../
    MLEndSND_Public/
        *.wav
    MLEndSND_Audio_Attributes.csv
    MLEndSND_Speakers_Demographics.csv
"""

from pathlib import Path

import click
import pandas as pd

from ertk.dataset import resample_audio, write_annotations, write_filelist


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the MLEnd dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list((input_dir / "MLEndSND_Public").glob("*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        paths = list(resample_dir.glob("*.wav"))
    write_filelist(paths, "files_all")

    annots = pd.read_csv(input_dir / "MLEndSND_Audio_Attributes.csv", dtype=str)
    annots.set_index("Public filename", inplace=True)
    spk_info = pd.read_csv(input_dir / "MLEndSND_Speakers_Demographics.csv", dtype=str)
    spk_info.set_index("Speaker", inplace=True)
    df = annots.join(spk_info, "Speaker")

    write_annotations(df["Intonation"], "label")
    write_annotations(df["Speaker"], "speaker")
    write_annotations(df["Numeral"], "numeral")
    write_annotations(df["Nationality"], "nationality")
    write_annotations(df["Language1"], "l1")
    write_annotations({p.stem: "en" for p in paths}, "language")


if __name__ == "__main__":
    main()
