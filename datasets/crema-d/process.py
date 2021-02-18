"""Process the raw CaFE dataset.

This assumes the file structure from the original compressed file:
/.../
    AudioMP3/
        *.mp3 [for 44.1 kHz]
    AudioWAV/
        *.wav [for 16 kHz]
    ...
"""

from pathlib import Path
from typing import Dict

import click
import pandas as pd

from emotion_recognition.dataset import write_filelist, write_labels
from emotion_recognition.stats import alpha
from emotion_recognition.utils import PathlibPath

emotion_map = {
    'A': 'anger',
    'D': 'disgust',
    'F': 'fear',
    'H': 'happiness',
    'S': 'sadness',
    'N': 'neutral',
}


def write_labelset(name: str, labels: Dict[str, str]):
    df = pd.DataFrame({'Name': labels.keys(), 'Emotion': labels.values()})
    df.to_csv('labels_{}.csv'.format(name), header=True, index=False)


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process CREMA-D dataset at location INPUT_DIR."""

    paths = list(input_dir.glob('AudioWAV/*.wav'))
    # 1076_MTI_SAD_XX has no audio signal
    write_filelist([p for p in paths if p.stem != '1076_MTI_SAD_XX'])
    write_labels({p.stem: emotion_map[p.stem[9]] for p in paths})

    summaryTable = pd.read_csv('processedResults/summaryTable.csv',
                               low_memory=False, index_col=0)
    summaryTable['ActedEmo'] = summaryTable['FileName'].apply(lambda x: x[9])

    for mode in ['VoiceVote', 'FaceVote', 'MultiModalVote']:
        # Proportion of majority vote equivalent to acted emotion
        accuracy = ((summaryTable[mode] == summaryTable['ActedEmo']).sum()
                    / len(summaryTable))
        print("Acted accuracy using {}: {:.3f}".format(mode, accuracy))
    print()

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

    resp_d = {'voice': voiceResp, 'face': faceResp, 'both': multiModalResp}
    for s, mode in resp_d.items():
        # Proportion of human responses equal to acted emotion
        accuracy = (mode['respEmo'] == mode['dispEmo']).sum() / len(mode)
        print("Human accuracy using {}: {:.3f}".format(s, accuracy))

        dataTable = (mode.set_index(['sessionNums', 'clipNum'])['respEmo']
                     .astype('category').cat.codes.unstack() + 1)
        dataTable[dataTable.isna()] = 0
        data = dataTable.astype(int).to_numpy()
        print("Krippendorf's alpha using {}: {:.3f}".format(s, alpha(data)))
        print()

    tabulatedVotes = pd.read_csv('processedResults/tabulatedVotes.csv',
                                 low_memory=False, index_col=0)
    tabulatedVotes['mode'] = tabulatedVotes.index // 100000
    tabulatedVotes['mode'] = tabulatedVotes['mode'].map(
        lambda x: [None, 'voice', 'face', 'both'][x])
    print("Average vote agreement per annotation mode:")
    print(tabulatedVotes.groupby('mode')['agreement'].describe())


if __name__ == "__main__":
    main()
