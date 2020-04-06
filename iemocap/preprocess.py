#!/usr/bin/python3

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

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
    ratings = []
    for filename in Path('annotations').glob('*.txt'):
        with open(filename) as fid:
            for line in fid:
                line = line.strip()
                match = regex.match(line)
                if match:
                    name = match.group(3)
                    dimensions[name] = [float(match.group(i))
                                        for i in {5, 6, 7}]
                    labels[name] = match.group(4)
                elif line.startswith('C'):
                    rater, annotations = line.strip().split(':')
                    annotations = annotations.strip().split(';')[:-1]
                    annotations = [x.strip() for x in annotations]
                    if not rater.startswith('C') or rater[2] in 'MF':
                        continue
                    ratings.append((name, rater, annotations[0]))
    ratings = pd.DataFrame(sorted(ratings), columns=['name', 'rater', 'label'])
    agreement = np.mean((4 - ratings.groupby('name')['label'].nunique()) / 3)
    print("Mean label agreement: {:.3f}".format(agreement))

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
