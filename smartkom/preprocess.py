#!/usr/bin/python3

import argparse
from pathlib import Path

import numpy as np
import soundfile

parser = argparse.ArgumentParser()
parser.add_argument(
    '--set', type=str, required=True,
    help="SmartKom set to process, one of {Home, Mobil, Public}",
)
parser.add_argument('--labels', help="File to write annotations to",
                    default='labels.txt', type=str)
parser.add_argument('--wav_out', help="Directory to individual turns",
                    default='wav_corpus', type=str)


def main():
    args = parser.parse_args()

    labels_file = open(args.labels, 'w')
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
        with open(annot_file) as fid:
            for line in fid:
                if line.startswith('USH:'):
                    _, start, dur, emo, *rest = line.strip().split()
                    emo = emo.replace('"', '')
                    ush_list.append((int(start), int(dur), emo))
                if line.startswith('TRN:'):
                    _, start, dur, words, turn = line.strip().split()
                    trn_list.append((int(start), int(dur), turn))

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
            emo = trn_map[i]
            out_file = Path(args.wav_out) / '{}_{}_{:03d}.wav'.format(
                sess, speaker, i)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(out_file, audio[start:end], sr)
            print('{}, {}'.format(out_file.stem, emo), file=labels_file)
    labels_file.close()


if __name__ == "__main__":
    main()
