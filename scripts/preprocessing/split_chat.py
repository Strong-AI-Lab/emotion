"""Splits a CHAT file into respective turns designated by timecodes."""

import re
from pathlib import Path
from typing import Tuple

import click
import soundfile
from emorec.utils import PathlibPath
from joblib import Parallel, delayed

REGEX = re.compile(
    r"^\*([A-Z]{3}):\t.*(?:\n\t.*)*[.?!](?: .*\x15(\d+)_(\d+)\x15)?$", re.MULTILINE
)


def process(path: Path, out_dir: Path, prefix: str = ""):
    audio, sr = soundfile.read(path)
    assert sr == 16000, "Sample rate must be 16000 Hz."

    cha_file = path.with_suffix(".cha")
    with open(cha_file) as fid:
        for i, match in enumerate(REGEX.finditer(fid.read())):
            if match[2] is None or match[3] is None:
                # This utterance doesn't have timecode info
                continue

            s_ms = int(match[2])
            e_ms = int(match[3])
            s_sam = int(s_ms * sr / 1000)
            e_sam = int(e_ms * sr / 1000)
            if s_sam > len(audio):
                print(f"WARNING: audio {path} shorter than expected.")
                break
            out_name = f"{prefix}{path.stem}_{i + 1:03d}_{match[1]}"
            split = audio[s_sam:e_sam]
            out_file = out_dir / (out_name + ".wav")
            soundfile.write(out_file, split, sr)


@click.command()
@click.argument("input", type=PathlibPath(exists=True), nargs=-1)
@click.argument("output", type=Path)
@click.option("--prefix", type=str, default="")
def main(input: Tuple[Path], output: Path, prefix: str):
    output.mkdir(parents=True, exist_ok=True)
    for path in input:
        print(f"Processing directory {path}")
        Parallel(n_jobs=-1, prefer="threads", verbose=1)(
            delayed(process)(p, output, prefix) for p in path.glob("**/*.wav")
        )


if __name__ == "__main__":
    main()
