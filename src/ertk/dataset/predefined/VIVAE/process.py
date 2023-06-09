"""Process the raw VIVAE dataset.

This assumes the file structure from the original data:
/.../
    full_set/
        *.wav
    core_set/
        *.wav
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the VIVAE dataset at location INPUT_DIR."""
    paths = list(input_dir.glob("full_set/*.wav"))
    resample_dir = Path("resampled")
    if resample:
        resample_audio(paths, resample_dir)
    write_filelist(resample_dir.glob("*.wav"), "files_all")
    core_set = [
        resample_dir / x.name
        for x in paths
        if (input_dir / "core_set" / x.name).exists()
    ]
    write_filelist(core_set, "files_core")

    labels = {}
    speakers = {}
    intensities = {}
    for path in paths:
        spk, emo, intensity, _ = path.stem.split("_")
        labels[path.stem] = emo
        speakers[path.stem] = spk
        intensities[path.stem] = intensity

    write_annotations(labels, "label")
    write_annotations(speakers, "speaker")
    write_annotations(intensities, "intensity")


if __name__ == "__main__":
    main()
