import re
import shutil
from pathlib import Path

import pandas as pd

MED_INT_REGEX = re.compile(r'^(\d\d?)([A-Za-z]+)$')
HIGH_INT_REGEX = re.compile(r'^[A-Za-z]+(\d+)([A-Za-z]+)\d?(?:[Hh]igh|Hi|H)$')
EMOTIONS = {
    'aff': 'affection',
    'amu': 'amusement',
    'ang': 'anger',
    'con': 'contempt',
    'disg': 'disgust',
    'dist': 'distress',
    'fea': 'fear',
    'gui': 'guilt',
    'hap': 'happiness',
    'int': 'interest',
    'lus': 'lust',
    'negs': 'negativesurprise',
    'negsurprise': 'negativesurprise',
    'neu': 'neutral',
    'nsur': 'negativesurprise',
    'nsu': 'negativesurprise',
    'poss': 'positivesurprise',
    'possurprise': 'positivesurprise',
    'psur': 'positivesurprise',
    'pri': 'pride',
    'rel': 'relief',
    'sad': 'sadness',
    'ser': 'serenity',
    'sha': 'shame',
    'sham': 'shame'
}


def get_speaker(s: str):
    try:
        return MED_INT_REGEX.match(s)[1]
    except TypeError:
        return HIGH_INT_REGEX.match(s)[1]


def get_emotion(s: str):
    try:
        emo = MED_INT_REGEX.match(s)[2].lower()
    except TypeError:
        emo = HIGH_INT_REGEX.match(s)[2].lower()
    return EMOTIONS.get(emo, emo)


def main():
    clip_info = pd.read_csv('veneccountryinfo.csv', index_col=0,
                            names=['Filename', 'Country', 'Original'])

    def get_ratings(country: str = 'USA') -> pd.DataFrame:
        ratings = pd.read_csv('CategoryRatings{}.csv'.format(country),
                              index_col=0, header=0)
        ratings.index.name = 'File'

        mode_count = ratings.apply(
            lambda x: x.value_counts().sort_index().iloc[-1], axis=1)
        ratings['Freq'] = ratings.max(1)
        ratings['Majority'] = ratings.columns[ratings.to_numpy().argmax(1)]
        ratings['Num_Majority'] = mode_count == 1

        agreement = ratings['Freq'].mean()
        print("Mean label agreement: {:.3f}".format(agreement))

        ratings = ratings.join(clip_info)
        ratings['Country'] = ratings['Country'].map(str.upper)
        # Correct the three errors where country is STR
        ratings.loc[ratings['Country'] == 'STR', 'Country'] = 'AUS'
        # Intensity is High or Medium
        ratings['Intensity'] = ratings['Original'].map(
            lambda x: Path(x).parts[0].split('_')[0])
        ratings['Speaker_No'] = ratings['Original'].map(
            lambda x: Path(x).stem).map(get_speaker).map(int)
        ratings['Speaker_ID'] = ratings['Country'] + '_' + ratings['Speaker_No'].map(
            '{:02d}'.format)
        ratings['Emotion'] = ratings['Original'].map(
            lambda x: Path(x).stem).map(get_emotion)
        del ratings['Original']
        del ratings['Country']
        del ratings['Speaker_No']

        # Remove 'neutral' since it has only 10 clips in medium
        # intensity in 3 countries.
        ratings = ratings[ratings['Emotion'] != 'neutral']
        keep_emotions = set(EMOTIONS.values()) - {'neutral'}

        # Keep only (country, speaker) pairs for which there are clips
        # for all emotions. This gives us around 2262 clips.
        ratings = ratings.groupby('Speaker_ID').filter(
            lambda x: set(x['Emotion']) == keep_emotions)
        ratings = ratings.sort_values(['Speaker_ID', 'Emotion'])
        return ratings

    ratings = get_ratings()
    ratings.index = ratings.index.map(lambda x: x[:-4])

    out_dir = Path('wav_corpus')
    out_dir.mkdir(exist_ok=True)
    for clip in ratings.index:
        name = clip + '_' + ratings.loc[clip, 'Speaker_ID'] + '.wav'
        shutil.copy('audio/' + clip + '.wav', out_dir / name)

    ratings.index = ratings.index.str.cat('_' + ratings['Speaker_ID'])
    # Acted labels
    ratings['Emotion'].to_csv('labels.csv')


if __name__ == "__main__":
    main()
