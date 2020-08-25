#!/usr/bin/python3

import argparse
from pathlib import Path

import pandas as pd

from emotion_recognition.stats import alpha

emotions = {
    'A': 'anger',
    'D': 'disgust',
    'F': 'fear',
    'H': 'happy',
    'S': 'sad',
    'N': 'neutral',
}

parser = argparse.ArgumentParser()
parser.add_argument('--classification', type=Path,
                    help="File to write classification annotations to.")
parser.add_argument('--wav_in', help="Directory storing WAV files.", type=Path)
parser.add_argument('--list_out', help="File to write filenames.", type=Path)


def main():
    args = parser.parse_args()

    finishedResponses = pd.read_csv('finishedResponses.csv', low_memory=False,
                                    index_col=0)
    finishedResponses['respLevel'] = pd.to_numeric(
        finishedResponses['respLevel'], errors='coerce')
    # Remove these two duplicates
    finishedResponses = finishedResponses.drop([137526, 312184],
                                               errors='ignore')
    uniqueIDs = (finishedResponses['sessionNums'] * 1000
                 + finishedResponses['queryType'] * 100
                 + finishedResponses['questNum'])

    finishedEmoResponses = pd.read_csv('finishedEmoResponses.csv',
                                       low_memory=False, index_col=0)
    finishedEmoResponses = finishedEmoResponses.query(
        'clipNum != 7443 and clipNum != 7444')

    distractedResponses = finishedEmoResponses.query('ttr > 10000')
    distractedIDs = (distractedResponses['sessionNums'] * 1000
                     + distractedResponses['queryType'] * 100
                     + distractedResponses['questNum'])
    goodResponses = finishedResponses[~uniqueIDs.isin(distractedIDs)]

    voiceResp = goodResponses.query('queryType == 1')
    faceResp = goodResponses.query('queryType == 2')
    multiModalResp = goodResponses.query('queryType == 3')

    for mode in [voiceResp, faceResp, multiModalResp]:
        accuracy = (mode['respEmo'] == mode['dispEmo']).sum() / len(mode)
        print("Human accuracy: {:.3f}".format(accuracy))

        dataTable = (mode.set_index(['sessionNums', 'clipNum'])['respEmo']
                     .astype('category').cat.codes.unstack() + 1)
        dataTable[dataTable.isna()] = 0
        data = dataTable.astype(int).to_numpy()
        print("Krippendorf's alpha: {:.3f}".format(alpha(data)))

    tabulatedVotes = pd.read_csv('processedResults/tabulatedVotes.csv',
                                 low_memory=False, index_col=0)
    tabulatedVotes['mode'] = tabulatedVotes.index // 100000
    print("Average vote agreement per annotation mode:")
    print(tabulatedVotes.groupby('mode')['agreement'].describe())

    summaryTable = pd.read_csv('processedResults/summaryTable.csv',
                               low_memory=False, index_col=0)
    summaryTable['ActedEmo'] = summaryTable['FileName'].apply(lambda x: x[9])

    for mode in ['VoiceVote', 'FaceVote', 'MultiModalVote']:
        accuracy = (summaryTable[mode] == summaryTable['ActedEmo'].sum()
                    / len(summaryTable))
        print("Acted accuracy using {}: {:.3f}".format(mode, accuracy))

    labels = dict(zip(summaryTable['FileName'], summaryTable['ActedEmo']))
    labels.pop('1076_MTI_SAD_XX')  # This one has no audio signal

    # valid = summaryTable['MultiModalVote'].isin(list('NHDFAS'))
    # multiModal = summaryTable[valid]
    # leftovers = summaryTable[~valid]

    # valid = leftovers['FaceVote'].isin(list('NHDFAS'))
    # face = leftovers[valid]
    # leftovers = leftovers[~valid]

    # valid = leftovers['VoiceVote'].isin(list('NHDFAS'))
    # voice = leftovers[valid]
    # leftovers = leftovers[~valid]

    # labels = dict(zip(multiModal['FileName'],
    #                   multiModal['MultiModalVote']))
    # labels.update(zip(face['FileName'],
    #                   face['FaceVote']))
    # labels.update(zip(voice['FileName'],
    #                   voice['VoiceVote']))

    if args.classification:
        with open(args.classification, 'w') as fid:
            print("Name,Emotion", file=fid)
            for name, emo in sorted(labels.items()):
                emo = emotions[emo]
                print('{},{}'.format(name, emo), file=fid)

    if args.wav_in and args.list_out:
        with open(args.list_out, 'w') as fid:
            for name, _ in sorted(labels.items()):
                if name not in labels:
                    continue
                src = args.wav_in / '{}.wav'.format(name)
                print(src.resolve(), file=fid)


if __name__ == "__main__":
    main()
