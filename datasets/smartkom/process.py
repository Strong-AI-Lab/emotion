import argparse
from pathlib import Path

import numpy as np
import soundfile
from nltk.corpus import stopwords

emotions = {
    'Neutral': 'neutral',
    'Freude_Erfolg': 'happiness',
    'Uberlegen_Nachdenken': 'pondering',
    'Ratlosigkeit': 'helplessness',
    'Arger_Miserfolg': 'anger',
    'Uberraschung_Verwunderung': 'surprise',
    'Restklasse': 'unknown'
}

stops = set(stopwords.words('german'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--set', type=str, required=True,
        help="SmartKom set to process, one of {Home, Mobil, Public}",
    )
    parser.add_argument('--wav_out', help="Directory to individual turns",
                        default='wav_corpus', type=Path)
    parser.add_argument('--transcripts', help="Transcripts file",
                        default='transcripts.csv', type=Path)
    args = parser.parse_args()

    labels_file = open('labels.csv', 'w')
    transcripts_file = open(args.transcripts, 'w')
    print('Name,Emotion', file=labels_file)
    print('Name,Transcript', file=transcripts_file)
    for sess_dir in sorted(Path('SK-' + args.set).glob('*')):
        print(sess_dir)
        sess = sess_dir.name
        wav_file = next(sess_dir.glob(sess + '?.wav'))
        audio, sr = soundfile.read(wav_file)
        annot_file = next(x for x in sess_dir.glob(sess + '_???.par')
                          if 'SMA' not in x.name)
        speaker = annot_file.stem[-3:]

        ush_list = []
        trn_list = []
        words = {}
        with open(annot_file) as fid:
            for line in fid:
                if line.startswith('USH:'):
                    _, start, dur, emo, *rest = line.strip().split()
                    emo = emo.replace('"', '')
                    ush_list.append((int(start), int(dur), emo))
                elif line.startswith('TRN:'):
                    _, start, dur, indices, turn = line.strip().split()
                    indices = [int(x) for x in indices.split(',')]
                    trn_list.append((int(start), int(dur), indices))
                elif line.startswith('ORT:'):
                    _, idx, word = line.strip().split()
                    idx = int(idx)
                    words[idx] = word

        trn_text = []
        for trn in trn_list:
            wordlist = [words[i] for i in trn[2]]
            wordlist = [x for x in wordlist if '<' not in x and '>' not in x]
            wordlist = [x for x in wordlist if x.lower() not in stops]
            s = ' '.join(wordlist)
            s = s.replace('"', r'\"')
            trn_text.append(s)

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
            if start == end:
                if start >= len(ush_list):
                    emo = ush_list[start - 1][2]
                else:
                    emo = ush_list[start][2]
                count = {emo: trn_ends[idx] - trn_starts[idx]}
            else:
                emo = ush_list[start][2]
                count = {emo: ush_ends[start] - trn_starts[idx]}
                for ivl in range(start + 1, end):
                    emo = ush_list[ivl][2]
                    count[emo] = (count.setdefault(emo, 0)
                                  + ush_ends[ivl]
                                  - ush_starts[ivl])
                if end >= len(ush_list):
                    emo = ush_list[-1][2]
                    count[emo] = (count.setdefault(emo, 0)
                                  + trn_ends[idx]
                                  - ush_ends[-1])
                else:
                    emo = ush_list[end][2]
                    count[emo] = (count.setdefault(emo, 0)
                                  + trn_ends[idx]
                                  - ush_starts[end])
            emotion = list(count.keys())[np.argmax(list(count.values()))]
            trn_map.append(emotion)

        for i, (start, end) in enumerate(zip(trn_starts, trn_ends)):
            name = '{}_{}_{:03d}'.format(sess, speaker, i)
            if args.wav_out:
                out_file = (args.wav_out / name).with_suffix('.wav')
                out_file.parent.mkdir(parents=True, exist_ok=True)
                soundfile.write(out_file, audio[start:end], sr)

            print('{},"{}"'.format(name, trn_text[i]), file=transcripts_file)

            emo = trn_map[i].replace('/', '_')
            emo = emotions[emo]
            print('{},{}'.format(name, emo), file=labels_file)
    labels_file.close()
    transcripts_file.close()


if __name__ == "__main__":
    main()
