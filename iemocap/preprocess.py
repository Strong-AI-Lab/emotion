#!/usr/bin/python3

import argparse
import re
from pathlib import Path

# [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
REGEX = (r'^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\w+)\t(\w+)\t'
         r'\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]$')

emotions = {
    'ang': 'anger',
    'hap': 'happiness',
    'sad': 'sadness',
    'neu': 'neutral',

    'dis': 'disgust',
    'exc': 'excitement',
    'fru': 'frustration',
    'fea': 'fear',
    'sur': 'surprise',
    'oth': 'other',
    'xxx': 'unknown'
}

parser = argparse.ArgumentParser()
parser.add_argument('--annotations', help="IEMOCAP annotations directory",
                    default='annotations', type=Path)
parser.add_argument('--regression', type=Path,
                    help="File to write regression annotations to")
parser.add_argument('--classification', type=Path,
                    help="File to write classification annotations to")
parser.add_argument('--wav_in', help="Directory storing WAV files", type=Path)
parser.add_argument('--list_out', help="File to write filenames", type=Path)


def main():
    args = parser.parse_args()

    regex = re.compile(REGEX)
    dimensions = {}
    labels = {}
    for filename in args.annotations.glob('*.txt'):
        with open(filename) as fid:
            for line in fid:
                match = regex.match(line.strip())
                if match:
                    dimensions[match.group(3)] = [float(match.group(i))
                                                  for i in {5, 6, 7}]
                    labels[match.group(3)] = match.group(4)

    if args.regression:
        with open(args.regression, 'w') as fid:
            print("Name,Valence,Activation,Dominance", file=fid)
            for name, (v, a, d) in sorted(dimensions.items()):
                print('{},{},{},{}'.format(name, v, a, d), file=fid)

    if args.classification:
        with open(args.classification, 'w') as fid:
            print("Name,Emotion", file=fid)
            for name, emo in sorted(labels.items()):
                if emo == 'exc':
                    # Merge happiness and excitement
                    emo = 'hap'
                if emo in emotions.keys():
                    emo = emotions[emo]
                print('{},{}'.format(name, emo), file=fid)

    if args.wav_in and args.list_out:
        with open(args.list_out, 'w') as fid:
            for name, emo in sorted(labels.items()):
                if emo not in ['ang', 'hap', 'neu', 'sad', 'exc']:
                    continue
                src = args.wav_in / '{}.wav'.format(name)
                print(src.resolve(), file=fid)


if __name__ == "__main__":
    main()
