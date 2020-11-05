#!/usr/bin/python3

"""Splits a CHAT file into respective turns designated by timecodes."""

import argparse
import re
from pathlib import Path

import soundfile
from joblib import Parallel, delayed

REGEX = re.compile(
    r'^\*([A-Z]{3}):\t.*(?:\n\t.*)*[.?!](?: \x15(\d+)_(\d+)\x15)?$',
    re.MULTILINE
)


def process(path: Path, out_dir: Path, prefix: str = ''):
    audio, sr = soundfile.read(path)
    assert sr == 16000, "Sample rate must be 16000 Hz."

    cha_file = path.with_suffix('.cha')
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
                print("WARNING: audio {} shorter than expected.".format(path))
                break
            out_name = '{}{}_{:03d}_{}'.format(prefix, path.stem, i + 1,
                                               match[1])
            split = audio[s_sam:e_sam]
            out_file = out_dir / (out_name + '.wav')
            soundfile.write(out_file, split, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, nargs='+',
                        help="Input director(y|ies).")
    parser.add_argument('--output', type=Path, required=True,
                        help="Output directory.")
    parser.add_argument('--prefix', type=str, default='', help="Name prefix.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.input:
        print("Processing directory {}".format(path))
        Parallel(n_jobs=-1, prefer='threads', verbose=1)(
            delayed(process)(p, args.output, args.prefix)
            for p in path.glob('**/*.wav')
        )


if __name__ == "__main__":
    main()
