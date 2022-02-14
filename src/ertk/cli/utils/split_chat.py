import re
import warnings
from pathlib import Path
from typing import Tuple

import click
import librosa
import soundfile
from joblib import delayed

from ertk.utils import TqdmParallel

REGEX = re.compile(
    r"^\*([A-Z]{3}):\t.*(?:\n\t.*)*[.?!](?: .*\x15(\d+)_(\d+)\x15)?$", re.MULTILINE
)


@click.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.argument("output", type=Path)
@click.option("--prefix", type=str, default="")
def main(input: Tuple[Path], output: Path, prefix: str):
    """Splits CHAT (.cha) files and associated audio into segments."""

    def process(path: Path):
        warnings.simplefilter("ignore", UserWarning)
        audio, _ = librosa.load(path, sr=16000, res_type="kaiser_fast")
        warnings.simplefilter("default", UserWarning)

        cha_file = path.with_suffix(".cha")
        with open(cha_file) as fid:
            for i, match in enumerate(REGEX.finditer(fid.read())):
                if match[2] is None or match[3] is None:
                    # This utterance doesn't have timecode info
                    continue

                s_sam = int(match[2]) * 16
                e_sam = int(match[3]) * 16
                if s_sam > len(audio):
                    print(f"WARNING: audio {path} shorter than expected.")
                    break
                out_name = f"{prefix}{path.stem}_{i + 1:03d}_{match[1]}"
                split = audio[s_sam:e_sam]
                out_file = output / (out_name + ".wav")
                soundfile.write(out_file, split, 16000)

    output.mkdir(parents=True, exist_ok=True)
    all_files = [p for path in input for p in path.glob("**/*.mp3")]
    TqdmParallel(total=len(all_files), desc="Processing CHAT files", n_jobs=-1)(
        map(delayed(process), all_files)
    )


if __name__ == "__main__":
    main()
