import argparse
import os
import re

import numpy as np
import PyWave

COMBINED_DIR = 'combined'
BITS_PER_SAMPLE = 16
SAMPLE_RATE = 16000

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", default="combined")
parser.add_argument("annotation_file", default="annot.txt")


def main():
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError("Directory doesn't exist: " + args.input_dir)

    annot_file = open(args.annotation_file, 'w')

    recordings = os.listdir(args.input_dir)
    for recording in sorted(recordings):
        recording_dir = os.path.join(args.input_dir, recording)
        if not os.path.isdir(recording_dir):
            continue

        with PyWave.Wave(os.path.join(recording_dir,
                                      'operator_audio.wav')) as w:
            operator_audio = w.read()
        with PyWave.Wave(os.path.join(recording_dir,
                                      'user_audio.wav')) as w:
            user_audio = w.read()
        with open(os.path.join(recording_dir, 'emotions.txt')) as fid:
            lines = [l.strip() for l in fid][1:]  # Ignore header line
            emotion_data = np.array([
                [float(x) for x in l.split()] for l in lines])

        user_turns = {}
        operator_turns = {}
        for d, f in [(user_turns, 'words_user.txt'),
                     (operator_turns, 'words_operator.txt')]:
            with open(os.path.join(recording_dir, f)) as fid:
                for line in fid:
                    if line.startswith('---'):
                        m = re.search(r'---recording.*turn ([0-9]+)---', line)
                        turn = int(m.group(1))
                    else:
                        m = re.search(r'([0-9]+) ([0-9]+) <?([A-Z\'?!]+)>?',
                                      line)
                        if m:
                            start = int(m.group(1)) * (SAMPLE_RATE // 1000)
                            end = int(m.group(2)) * (SAMPLE_RATE // 1000)
                            word = m.group(3)
                            if turn not in d:
                                d[turn] = []
                            d[turn].append((start, end, word))

        for d, p, audio in [(user_turns, 'u', user_audio),
                            (operator_turns, 'o', operator_audio)]:
            out_dir = os.path.join(recording_dir, 'turns')
            os.makedirs(out_dir, exist_ok=True)
            for turn, words in sorted(d.items()):
                start = words[0][0]
                end = words[-1][1]

                name = "{:02d}_{}_{:03d}".format(int(recording), p, turn)
                filename = name + '.wav'
                w = PyWave.Wave(os.path.join(recording_dir, 'turns', filename),
                                mode='w',
                                channels=1,
                                frequency=SAMPLE_RATE,
                                bps=BITS_PER_SAMPLE)
                w.write(audio[
                    start * (BITS_PER_SAMPLE // 8):end * (BITS_PER_SAMPLE // 8)
                ])

                if p == 'u':
                    start_idx, end_idx = np.searchsorted(
                        emotion_data[:, 0],
                        [start / SAMPLE_RATE, end / SAMPLE_RATE]
                    )
                    if start_idx != end_idx:
                        mean_emotions = emotion_data[
                            start_idx:end_idx, 1:].mean(0)
                        print(
                            "{}, A: {:.4f}, E: {:.4f}, P: {:.4f}, V: {:.4f}"
                            .format(name, *mean_emotions),
                            file=annot_file
                        )
    annot_file.close()


if __name__ == "__main__":
    main()
