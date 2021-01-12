import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from emotion_recognition.stats import alpha

REGEX = re.compile(r'^UTD-IMPROV-([A-Z0-9-]+)\.avi; ([A-Z]); A:(\d+\.\d+|NaN); V:(\d+\.\d+|NaN); D:(\d+\.\d+|NaN) ; N:(\d+\.\d+|NaN);$')  # noqa

emotions = {
    'A': 'anger',
    'H': 'happiness',
    'S': 'sadness',
    'N': 'neutral',

    'O': 'other',
    'X': 'unknown'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=Path, default='wav_corpus',
                        help="Directory storing WAV files")
    args = parser.parse_args()

    dimensions = {}
    labels = {}
    ratings = []
    with open('Evaluation.txt') as fid:
        name = ''
        for line in fid:
            line = line.strip()
            match = REGEX.match(line)
            if match:
                name = match.group(1)
                labels[name] = match.group(2)
                dimensions[name] = list(map(float, match.group(3, 4, 5, 6)))
            elif line != '':
                rater, label, *_ = line.split(';')
                label = label.strip()[0]
                ratings.append((name, rater, label))
    ratings = pd.DataFrame(sorted(ratings), columns=['name', 'rater', 'label'])
    ratings = ratings.drop_duplicates(['name', 'rater'])

    num_ratings = ratings.groupby('name').size().to_frame('total')
    label_count = ratings.groupby(['name', 'label']).size().to_frame('freq')
    # Count of majority label per utterance
    mode_count = (
        label_count.reset_index().sort_values('freq', ascending=False)
        .drop_duplicates(subset='name').set_index('name')
        .join(num_ratings).sort_index()
    )

    # Include only names with a label which is strictly a plurality
    mode_count = mode_count[
        ratings.groupby('name')['label'].agg(lambda x: ''.join(x.mode()))
        .isin(set('NASH')).sort_index()
    ]
    # Acted label
    mode_count['acted'] = [x[3] for x in mode_count.index]

    # Agreement is mean proportion of labels which are plurality label
    agreement = np.mean(mode_count['freq'] / mode_count['total'])
    print("Mean label agreement: {:.3f}".format(agreement))

    # Agreement with acted label
    agreement = ((mode_count['label'] == mode_count['acted']).sum()
                 / len(mode_count))
    print("Acted agreement: {:.3f}".format(agreement))

    clips = ratings.join(mode_count['label'], 'name', rsuffix='_vote',
                         how='inner')
    accuracy = (clips['label'] == clips['label_vote']).sum() / len(clips)
    print("Human accuracy: {:.3f}".format(accuracy))

    # Simple way to get int matrix of labels for raters x clips
    data = (ratings.set_index(['rater', 'name'])['label'].astype('category')
            .cat.codes.unstack() + 1)
    data[data.isna()] = 0
    data = data.astype(int).to_numpy()
    print("Krippendorf's alpha: {:.3f}".format(alpha(data)))

    emo_dist = Counter(labels.values())
    print("Emotion distribution:")
    for emo, count in emo_dist.items():
        print("{:<10s}: {:d}".format(emotions[emo], count))

    # Aggregated dimensional annotations per utterance
    with open('valence.csv', 'w') as vfile, \
            open('activation.csv', 'w') as afile, \
            open('dominance.csv', 'w') as dfile, \
            open('naturalness.csv', 'w') as nfile:
        print("Name,Valence", file=vfile)
        print("Name,Activation", file=afile)
        print("Name,Dominance", file=dfile)
        print("Name,Naturalness", file=nfile)
        for name, (a, v, d, n) in sorted(dimensions.items()):
            print('{},{}'.format(name, v), file=vfile)
            print('{},{}'.format(name, a), file=afile)
            print('{},{}'.format(name, d), file=dfile)
            print('{},{}'.format(name, n), file=nfile)

    # Emotion labels per utterance
    with open('labels.csv', 'w') as fid:
        print("Name,Emotion", file=fid)
        for name, emo in sorted(labels.items()):
            emo = emotions[emo]
            print('{},{}'.format(name, emo), file=fid)

    # Audio filepaths
    with open('files.txt', 'w') as fid:
        for name, emo in sorted(labels.items()):
            if emo not in 'AHSN':
                continue
            src = args.audio / '{}.wav'.format(name)
            print(src.resolve(), file=fid)


if __name__ == "__main__":
    main()
