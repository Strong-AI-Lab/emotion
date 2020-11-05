#!/usr/bin/python3

"""Selects speech clips of a given length."""

import argparse
from pathlib import Path

import soundfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help="Input directory.")
    parser.add_argument(
        '--minlength', type=float, default=5,
        help="Minimum length (in seconds) of resulting speech clips."
    )
    parser.add_argument(
        '--maxlength', type=float, default=15,
        help="Maximum length (in seconds) of resulting speech clips."
    )
    parser.add_argument('--output', type=Path, required=True,
                        help="Output file.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as fid:
        for path in sorted(args.input.glob('*.wav')):
            length = soundfile.info(path).duration
            if args.minlength < length < args.maxlength:
                print(path.absolute(), file=fid)


if __name__ == "__main__":
    main()
