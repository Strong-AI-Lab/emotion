"""Process the raw JL dataset.

This assumes the file structure from the original compressed file:
/.../
    Raw JL Corpus (unchecked ...)/
        JL(wav+txt)/
            *.wav
        ...
    ...
"""

import re
from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist

REGEX = re.compile(r"^(?:fe)?male[12]_([a-z]+)_\d+[ab]_[12]$")

emotion_map = {
    "angry": "anger",
    "sad": "sadness",
    "neutral": "neutral",
    "happy": "happiness",
    "excited": "excitedness",
    "anxious": "anxiety",
    "apologetic": "apologetic",
    "assertive": "assertive",
    "concerned": "concern",
    "encouraging": "encouraging",
}

secondary_emotions = ["anxious", "apologetic", "assertive", "concerned", "encouraging"]


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the JL dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(
        input_dir.glob("Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/*.wav")
    )

    transcripts = {}
    for p in paths:
        trn_file = p.with_suffix(".txt")
        with open(trn_file, "r") as fid:
            transcripts[p.stem] = fid.read().strip()
    write_annotations(transcripts, "transcript")

    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        paths = list(resample_dir.glob("*.wav"))
    write_filelist([p for p in paths], "files_all")
    write_filelist(
        [p for p in paths if REGEX.match(p.stem).group(1) not in secondary_emotions],
        "files_primary",
    )
    write_filelist(
        [p for p in paths if REGEX.match(p.stem).group(1) in secondary_emotions],
        "files_secondary",
    )

    write_annotations(
        {p.stem: emotion_map.get(REGEX.match(p.stem).group(1)) for p in paths},
        "label",
    )
    speaker_dict = {p.stem: p.stem[: p.stem.find("_")] for p in paths}
    write_annotations(speaker_dict, "speaker")
    write_annotations({k: v[0].upper() for k, v in speaker_dict.items()}, "gender")
    write_annotations({p.stem: "en" for p in paths}, "language")
    write_annotations({p.stem: "nz" for p in paths}, "country")
    write_annotations({p.stem: p.stem[-4:-2] for p in paths}, "sentence")


if __name__ == "__main__":
    main()
