from pathlib import Path

import click
import soundfile
from tqdm import tqdm

from ertk.dataset import get_audio_paths
from ertk.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True))
@click.argument("output", type=Path)
@click.option(
    "--minlength",
    type=float,
    default=5,
    help="Minimum length (in seconds) of resulting speech clips.",
)
@click.option(
    "--maxlength",
    type=float,
    default=5,
    help="Maximum length (in seconds) of resulting speech clips.",
)
def main(input: Path, output: Path, minlength: float, maxlength: float):
    """Selects speech clips in a given length range."""
    clips = []
    if input.is_dir():
        paths = sorted(input.glob("**/*.wav"))
    else:
        paths = get_audio_paths(input)
    print(f"Found {len(paths)} files total.")
    for path in tqdm(paths, desc="Processing clips", disable=None):
        length = soundfile.info(path).duration
        if minlength < length < maxlength:
            clips.append(str(path))
    print(f"Found {len(clips)} valid clips.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fid:
        fid.write("\n".join(clips) + "\n")


if __name__ == "__main__":
    main()
