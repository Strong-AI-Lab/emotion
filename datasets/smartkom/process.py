"""Process the raw SmartKom dataset.

This assumes the file structure from the original compressed file:
/.../
    SK-Public/
        w001_pk/
            w001_pkd.wav
            w001_pk_AAA.par
            ...
        ...
    ...
"""

from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pandas as pd
import soundfile
from emorec.dataset import write_filelist, write_labels
from emorec.utils import PathlibPath
from joblib import Parallel, delayed
from nltk.corpus import stopwords

emotion_map = {
    'Neutral': 'neutral',
    'Freude/Erfolg': 'happiness',
    'Uberlegen/Nachdenken': 'pondering',
    'Ratlosigkeit': 'helplessness',
    'Arger/Miserfolg': 'anger',
    'Uberraschung/Verwunderung': 'surprise',
    'Restklasse': 'unknown'
}

stops = set(stopwords.words('german'))
audio_dir = Path('audio')


def process_sess(sess_dir: Path):
    emotions = {}
    transcripts = {}
    wav_file = next(sess_dir.glob(sess_dir.name + '?.wav'))
    audio, sr = soundfile.read(wav_file)

    annot_file = next(x for x in sess_dir.glob(sess_dir.name + '_???.par')
                      if 'SMA' not in x.name)
    ush_list = []
    trn_list = []
    words = {}
    with open(annot_file) as fid:
        for line in fid:
            if line.startswith('USH:'):
                # User state segments and labels
                _, start, dur, emo, *rest = line.strip().split()
                emo = emo.replace('"', '')
                ush_list.append((int(start), int(dur), emo))
            elif line.startswith('TRN:'):
                # Turn segments
                _, start, dur, indices, turn = line.strip().split()
                indices = [int(x) for x in indices.split(',')]
                trn_list.append((int(start), int(dur), indices))
            elif line.startswith('ORT:'):
                # Orthographic transcription
                _, idx, word = line.strip().split()
                idx = int(idx)
                words[idx] = word
    if len(ush_list) == 0 or len(trn_list) == 0:
        print("Missing USH or TRN for {}".format(sess_dir.name))
        return {}, {}

    trn_text = []
    for trn in trn_list:
        wordlist = [words[i] for i in trn[2]]
        wordlist = [x for x in wordlist if '<' not in x and '>' not in x]
        wordlist = [x for x in wordlist if x.lower() not in stops]
        s = ' '.join(wordlist)
        s = s.replace('"', r'\"')
        trn_text.append(s)

    # Get overlapping USH segments for each turn so that we can get
    # the majority label per turn
    ush_starts = np.array([x[0] for x in ush_list])
    ush_ends = np.array([x[0] + x[1] + 1 for x in ush_list])
    trn_starts = np.array([x[0] for x in trn_list])
    trn_ends = np.array([x[0] + x[1] + 1 for x in trn_list])
    start_intervals = np.searchsorted(ush_ends, trn_starts)
    end_intervals = np.searchsorted(ush_ends, trn_ends)
    trn_map = []
    for idx in range(len(trn_list)):
        start = start_intervals[idx]
        end = end_intervals[idx]
        count = defaultdict(int)
        if start == end:
            offset = int(start >= len(ush_list))
            emo = ush_list[start - offset][2]
            count[emo] = trn_ends[idx] - trn_starts[idx]
        else:
            emo = ush_list[start][2]
            count[emo] = ush_ends[start] - trn_starts[idx]
            for ivl in range(start + 1, end):
                emo = ush_list[ivl][2]
                count[emo] += ush_ends[ivl] - ush_starts[ivl]
            if end >= len(ush_list):
                emo = ush_list[-1][2]
                count[emo] += trn_ends[idx] - ush_ends[-1]
            else:
                emo = ush_list[end][2]
                count[emo] += trn_ends[idx] - ush_starts[end]
        emotion = list(count.keys())[np.argmax(count.values())]
        trn_map.append(emotion)

    for i, (start, end) in enumerate(zip(trn_starts, trn_ends)):
        name = '{}_{:03d}'.format(annot_file.stem, i)
        soundfile.write(audio_dir / (name + '.wav'), audio[start:end], sr)
        emotions[name] = emotion_map[trn_map[i]]
        transcripts[name] = trn_text[i]
    return emotions, transcripts


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the SmartKom dataset at location INPUT_DIR. Split audio
    into turns and select majority emotion as label.
    """

    emotions = {}
    transcripts = {}
    audio_dir.mkdir(parents=True, exist_ok=True)
    r = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_sess)(s) for s in (input_dir / 'SK-Public').glob('*')
        if s.is_dir()
    )
    for emo, trn in r:
        emotions.update(emo)
        transcripts.update(trn)
    write_filelist(audio_dir.glob('*.wav'))
    write_labels(emotions)
    df = pd.DataFrame.from_dict(transcripts, orient='index',
                                columns=['Transcripts'])
    df.index.name = 'Name'
    df.to_csv('transcripts.csv')


if __name__ == "__main__":
    main()
