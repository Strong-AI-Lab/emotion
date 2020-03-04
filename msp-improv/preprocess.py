#!/usr/bin/python3

import argparse
import re
from pathlib import Path

DIGIT = r'\d+\.\d+|NaN'
REGEX = (r'^UTD-IMPROV-([A-Z0-9-]+)\.avi; ([A-Z]); '
         r'A:({0}); V:({0}); D:({0}) ; N:({0});$'.format(DIGIT))

parser = argparse.ArgumentParser()
parser.add_argument('annot', help="MSP-IMPROV annotations file",
                    default='Evalution.txt', type=str)
parser.add_argument('--regression', type=str,
                    help="File to write regression annotations to")
parser.add_argument('--classification', type=str,
                    help="File to write classification annotations to")
parser.add_argument('--wav_in', help="Directory storing WAV files", type=str)
parser.add_argument('--list_out', help="File to write filenames", type=str)


def main():
    args = parser.parse_args()

    regex = re.compile(REGEX)
    dimensions = {}
    labels = {}
    with open(args.annot) as fid:
        for line in fid:
            match = regex.match(line.strip())
            if match:
                dimensions[match.group(1)] = [match.group(i)
                                              for i in [3, 4, 5, 6]]
                labels[match.group(1)] = match.group(2)

    if args.regression:
        with open(args.regression, 'w') as fid:
            for name, (a, v, d, n) in sorted(dimensions.items()):
                print('{}, A: {}, V: {}, D: {}, N: {}'.format(
                    name, a, v, d, n), file=fid)

    if args.classification:
        with open(args.classification, 'w') as fid:
            for name, emo in sorted(labels.items()):
                print('{}, {}'.format(name, emo), file=fid)

    if args.wav_in and args.list_out:
        with open(args.list_out, 'w') as fid:
            for name, emo in sorted(labels.items()):
                if emo not in 'AHSN':
                    continue
                src = Path(args.wav_in) / '{}.wav'.format(name)
                print(Path(src).resolve(), file=fid)


if __name__ == "__main__":
    main()
