"""Process the raw eNTERFACE dataset.

This assumes the file structure from the original compressed file:
/.../
    Session1/
        sentences/
            wav/
                Ses01F_impro01/
                    *.wav
                ...
            ...
        ...
    ...
"""

import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
from emorec.dataset import (resample_audio, write_filelist,
                                         write_labels)
from emorec.stats import alpha
from emorec.utils import PathlibPath

# [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
REGEX = re.compile(r'^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\w+)\t(\w+)\t\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]$')  # noqa

emotion_map = {
    'ang': 'anger',
    'hap': 'happiness',
    'sad': 'sadness',
    'neu': 'neutral',
    # Excitement is usually mapped to happiness
    'exc': 'happiness',
    'dis': 'disgust',
    'fru': 'frustration',
    'fea': 'fear',
    'sur': 'surprise',
    'oth': 'other',
    'xxx': 'unknown'
}

unused_emotions = ['dis', 'fru', 'fea', 'sur', 'oth', 'xxx']


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the IEMOCAP dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob('Session?/sentences/wav/*/*.wav'))
    resample_dir = Path('resampled')
    resample_audio(paths, resample_dir)

    dimensions = {}
    labels = {}
    ratings = []
    for filename in input_dir.glob('Session?/dialog/EmoEvaluation/*.txt'):
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
    write_filelist([p for p in resample_dir.glob('*.wav')
                    if labels[p.stem] not in unused_emotions])
    write_labels({n: emotion_map[labels[n]] for n in labels})

    # Aggregated dimensional annotations per utterance
    df = pd.DataFrame.from_dict(dimensions, orient='index',
                                columns=['Valence', 'Activation', 'Dominance'])
    df.index.name = 'Name'
    for dim in ['Valence', 'Activation', 'Dominance']:
        df[dim].to_csv(dim.lower() + '.csv', index=True, header=True)
        print("Wrote CSV to {}".format(dim.lower() + '.csv'))

    # Ratings analysis
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


if __name__ == "__main__":
    main()
