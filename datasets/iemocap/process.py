import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from emotion_recognition.stats import alpha

# [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
REGEX = re.compile(r'^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\w+)\t(\w+)\t\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]$')  # noqa

emotions = {
    'ang': 'anger',
    'hap': 'happiness',
    'sad': 'sadness',
    'neu': 'neutral',

    # Generally unused for discrete emoton classification
    'dis': 'disgust',
    'exc': 'excitement',
    'fru': 'frustration',
    'fea': 'fear',
    'sur': 'surprise',
    'oth': 'other',
    'xxx': 'unknown'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=Path, default='wav_corpus',
                        help="Directory storing WAV files")
    args = parser.parse_args()

    dimensions = {}
    labels = {}
    ratings = []
    for filename in Path('annotations').glob('*.txt'):
        with open(filename) as fid:
            name = ''
            for line in fid:
                line = line.strip()
                match = REGEX.match(line)
                if match:
                    name = match.group(3)
                    labels[name] = match.group(4)
                    dimensions[name] = list(map(float, match.group(5, 6, 7)))
                elif line.startswith('C'):
                    # C is classification
                    rater, annotations = line.strip().split(':')
                    rater = rater.strip().split('-')[1]
                    if rater[0] in 'MF':
                        # M or F refer to self-evalulation
                        continue
                    *annotations, comments = annotations.strip().split(';')
                    label = annotations[0].strip()
                    ratings.append((name, rater, label))
    ratings = pd.DataFrame(sorted(ratings), columns=['name', 'rater', 'label'])
    # There are 3 (non-self) raters per utterance. Agreement is the
    # proportion of labels which are the same. This formula only works
    # for 3 raters.
    agreement = np.mean((4 - ratings.groupby('name')['label'].nunique()) / 3)
    print("Mean label agreement: {:.3f}".format(agreement))

    # Simple way to get int matrix of labels for raters x clips
    data = (ratings.set_index(['rater', 'name'])['label'].astype('category')
            .cat.codes.unstack() + 1)
    data[data.isna()] = 0
    data = data.astype(int).to_numpy()
    print("Krippendorf's alpha: {:.3f}".format(alpha(data)))

    emo_dist = Counter(labels.values())
    print("Emotion distribution:")
    for emo, count in emo_dist.items():
        print("{:<12s}: {:d}".format(emotions[emo], count))

    return

    # Aggregated dimensional annotations per utterance
    with open('valence.csv', 'w') as vfile, \
            open('activation.csv', 'w') as afile, \
            open('dominance.csv', 'w') as dfile:
        print("Name,Valence", file=vfile)
        print("Name,Activation", file=afile)
        print("Name,Dominance", file=dfile)
        for name, (v, a, d) in sorted(dimensions.items()):
            print('{},{}'.format(name, v), file=vfile)
            print('{},{}'.format(name, a), file=afile)
            print('{},{}'.format(name, d), file=dfile)

    # Emotion labels per utterance
    with open('labels.csv', 'w') as fid:
        print("Name,Emotion", file=fid)
        for name, emo in sorted(labels.items()):
            # Merge happiness and excitement
            emo = 'happiness' if emo == 'exc' else emotions[emo]
            print('{},{}'.format(name, emo), file=fid)

    # Audio filepaths
    with open('files.txt', 'w') as fid:
        for name, emo in sorted(labels.items()):
            if emo not in ['ang', 'hap', 'neu', 'sad', 'exc']:
                continue
            src = args.audio / '{}.wav'.format(name)
            print(src.resolve(), file=fid)


if __name__ == "__main__":
    main()
