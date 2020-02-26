#!/usr/bin/python3

import argparse
import re
from pathlib import Path

# [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
REGEX = (r'^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\w+)\t(\w+)\t'
         r'\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]$')

parser = argparse.ArgumentParser()
parser.add_argument('dir', help="IEMOCAP annotations directory",
                    default='annotations', type=str)
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
    for filename in Path(args.dir).glob('*.txt'):
        with open(filename) as fid:
            for line in fid:
                line = line.strip()
                match = regex.match(line)
                if match:
                    dimensions[match.group(3)] = (match.group(5),
                                                  match.group(6),
                                                  match.group(7))
                    labels[match.group(3)] = match.group(4)

    if args.regression:
        with open(args.regression, 'w') as fid:
            for name, (v, a, d) in dimensions.items():
                print('{}, V: {}, A: {}, D: {}'.format(name, v, a, d),
                      file=fid)

    if args.classification:
        with open(args.classification, 'w') as fid:
            for name, emo in labels.items():
                if emo == 'exc':
                    # Merge happiness and excitement
                    emo = 'hap'
                print('{}, {}'.format(name, emo), file=fid)

    if args.wav_in and args.list_out:
        with open(args.list_out, 'w') as fid:
            for name, emo in labels.items():
                if emo not in ['ang', 'hap', 'neu', 'sad', 'exc']:
                    continue
                src = Path(args.wav_in) / '{}.wav'.format(name)
                print(Path(src).resolve(), file=fid)


if __name__ == "__main__":
    main()
