"""Selects speech clips in a given length range."""

import argparse
from pathlib import Path

import librosa
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
    parser.add_argument('output', type=Path, help="Output file.")
    args = parser.parse_args()

    clips = []
    for path in sorted(args.input.glob('*.wav')):
        length = soundfile.info(path).duration
        if args.minlength < length < args.maxlength:
            clips.append(str(path.absolute()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as fid:
        fid.write('\n'.join(clips) + '\n')


if __name__ == "__main__":
    main()
