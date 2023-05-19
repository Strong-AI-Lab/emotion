"""Process the raw ShEMO dataset.

This assumes the file structure from the original compressed file:
/.../
    male/
        *.wav
    female/
    transcript/
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
    name2path = {p.stem: p for p in paths}

    write_annotations({n: emotion_map[n[3]] for n in name2path}, "label")
    speaker_dict = {n: n[:3] for n in name2path}
    write_annotations(speaker_dict, "speaker")
    write_annotations({k: v[0] for k, v in speaker_dict.items()}, "gender")
    write_annotations({n: "fa" for n in name2path}, "language")
    write_annotations({n: "ir" for n in name2path}, "country")

    transcripts = {}
    for trn in (input_dir / "transcript/final text").glob("*.ort"):
        with open(trn, encoding="utf_8_sig") as fid:
            transcripts[trn.stem] = (
                fid.read().strip().replace("\u200e", "").replace("\u200f", "")
            )
    write_annotations(transcripts, "transcript")
    missing = set(name2path) - transcripts.keys()
    write_filelist([name2path[x] for x in missing], "missing_transcripts")


if __name__ == "__main__":
    main()
