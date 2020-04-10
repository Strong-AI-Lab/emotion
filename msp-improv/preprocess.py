#!/usr/bin/python3

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from emotion_recognition.stats import alpha

DIGIT = r'\d+\.\d+|NaN'
REGEX = (r'^UTD-IMPROV-([A-Z0-9-]+)\.avi; ([A-Z]); '
         r'A:({0}); V:({0}); D:({0}) ; N:({0});$'.format(DIGIT))

emotions = {
    'A': 'angry',
    'H': 'happy',
    'S': 'sad',
    'N': 'neutral',

    'O': 'other',
    'X': 'unknown'
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
    with open('Evaluation.txt') as fid:
        for line in fid:
            line = line.strip()
            match = regex.match(line)
            if match:
                name = match.group(1)
                dimensions[name] = [float(match.group(i))
                                    for i in [3, 4, 5, 6]]
                labels[name] = match.group(2)
            elif line != '':
                rater, label, *rest = line.split(';')
                label = label.strip()[0]
                ratings.append((name, rater, label))
    ratings = pd.DataFrame(sorted(ratings), columns=['name', 'rater', 'label'])
    ratings = ratings.drop_duplicates(['name', 'rater'])

    include = (
        ratings.groupby('name')['label'].agg(lambda x: ''.join(x.mode()))
        .isin(list('NASH')).sort_index()
    )

    mode_count = (
        ratings.groupby(['name', 'label']).size().to_frame('freq')
        .reset_index().sort_values('freq', ascending=False)
        .drop_duplicates(subset='name').set_index('name')
        .join(ratings.groupby('name').size().to_frame('total')).sort_index()
    )
    mode_count = mode_count[include]
    mode_count['true'] = [x[3] for x in mode_count.index]

    agreement = np.mean(mode_count['freq'] / mode_count['total'])
    print("Label agreement: {:.3f}".format(agreement))

    agreement = ((mode_count['label'] == mode_count['true']).sum()
                 / len(mode_count))
    print("Acted agreement: {:.3f}".format(agreement))

    clips = ratings.join(mode_count['label'], 'name', rsuffix='_vote',
                         how='inner')
    accuracy = (clips['label'] == clips['label_vote']).sum() / len(clips)
    print("Human accuracy: {:.3f}".format(accuracy))

    data = (ratings.set_index(['rater', 'name'])['label'].astype('category')
            .cat.codes.unstack() + 1)
    data[data.isna()] = 0
    data = data.astype(int).to_numpy()
    print("Krippendorf's alpha: {:.3f}".format(alpha(data)))

    emo_dist = Counter(labels.values())
    print("Emotion distribution:")
    for emotion, count in emo_dist.items():
        print("{:<10s}: {:d}".format(emotion, count))

    if args.regression:
        with open(args.regression, 'w') as fid:
            print("Name,Valence,Activation,Dominance,Naturalness", file=fid)
            for name, (a, v, d, n) in sorted(dimensions.items()):
                print('{},{},{},{},{}'.format(name, v, a, d, n), file=fid)

    if args.classification:
        with open(args.classification, 'w') as fid:
            print("Name,Emotion", file=fid)
            for name, emo in sorted(labels.items()):
                emo = emotions[emo]
                print('{},{}'.format(name, emo), file=fid)

    if args.wav_in and args.list_out:
        with open(args.list_out, 'w') as fid:
            for name, emo in sorted(labels.items()):
                if emo not in 'AHSN':
                    continue
                src = args.wav_in / '{}.wav'.format(name)
                print(src.resolve(), file=fid)


if __name__ == "__main__":
    main()
