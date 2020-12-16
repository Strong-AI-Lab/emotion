import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from emotion_recognition.stats import alpha

emotion_map = {
    'A': 'anger',
    'D': 'disgust',
    'F': 'fear',
    'H': 'happy',
    'S': 'sad',
    'N': 'neutral',
}


def write_labelset(name: str, labels: Dict[str, str]):
    with open('labels_{}.csv'.format(name), 'w') as label_file:
        label_file.write("Name,Emotion\n")
        for name, emo in labels.items():
            if name == '1076_MTI_SAD_XX':
                # This one has no audio signal
                continue
            label_file.write("{},{}\n".format(name, emotion_map[name[9]]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, help="Location of CREMA-D data.")
    args = parser.parse_args()

    files = sorted(args.input.glob('AudioWAV/*.wav'))
    # Write default clipset and annotationset using acted emotion
    with open('clips2.csv', 'w') as clip_file, \
            open('labels2.csv', 'w') as label_file:
        clip_file.write("Name,Path,Speaker\n")
        label_file.write("Name,Emotion\n")
        for path in files:
            name = path.stem
            if name == '1076_MTI_SAD_XX':
                # This one has no audio signal
                continue
            speaker = name[:4]
            clip_file.write("{0},AudioWAV/{0}.wav,{1}\n".format(name, speaker))
            label_file.write("{},{}\n".format(name, emotion_map[name[9]]))

    summaryTable = pd.read_csv('processedResults/summaryTable.csv',
                               low_memory=False, index_col=0)
    summaryTable['ActedEmo'] = summaryTable['FileName'].apply(lambda x: x[9])

    for mode in ['VoiceVote', 'FaceVote', 'MultiModalVote']:
        # Proportion of majority vote equivalent to acted emotion
        accuracy = (summaryTable[mode] == summaryTable['ActedEmo'].sum()
                    / len(summaryTable))
        print("Acted accuracy using {}: {:.3f}".format(mode, accuracy))

    # Majority vote annotations from other modalities
    valid = summaryTable['MultiModalVote'].isin(list('NHDFAS'))
    multiModal = summaryTable[valid]
    labels = dict(zip(multiModal['FileName'],
                      multiModal['MultiModalVote']))
    write_labelset('multimodal', labels)

    valid = summaryTable['FaceVote'].isin(list('NHDFAS'))
    face = summaryTable[valid]
    labels = dict(zip(face['FileName'],
                      face['FaceVote']))
    write_labelset('face', labels)

    valid = summaryTable['VoiceVote'].isin(list('NHDFAS'))
    voice = summaryTable[valid]
    labels = dict(zip(voice['FileName'],
                      voice['VoiceVote']))
    write_labelset('voice', labels)

    finishedResponses = pd.read_csv('finishedResponses.csv', low_memory=False,
                                    index_col=0)
    finishedResponses['respLevel'] = pd.to_numeric(
        finishedResponses['respLevel'], errors='coerce')
    # Remove these two duplicates
    finishedResponses = finishedResponses.drop([137526, 312184],
                                               errors='ignore')

    finishedEmoResponses = pd.read_csv('finishedEmoResponses.csv',
                                       low_memory=False, index_col=0)
    finishedEmoResponses = finishedEmoResponses.query(
        'clipNum != 7443 and clipNum != 7444')
    distractedResponses = finishedEmoResponses.query('ttr > 10000')

    uniqueIDs = (finishedResponses['sessionNums'] * 1000
                 + finishedResponses['queryType'] * 100
                 + finishedResponses['questNum'])
    distractedIDs = (distractedResponses['sessionNums'] * 1000
                     + distractedResponses['queryType'] * 100
                     + distractedResponses['questNum'])
    # Get all annotations not defined to be distracted
    goodResponses = finishedResponses[~uniqueIDs.isin(distractedIDs)]

    # Responses based on different modalities
    voiceResp = goodResponses.query('queryType == 1')
    faceResp = goodResponses.query('queryType == 2')
    multiModalResp = goodResponses.query('queryType == 3')

    for mode in [voiceResp, faceResp, multiModalResp]:
        # Proportion of human responses equal to acted emotion
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


if __name__ == "__main__":
    main()
