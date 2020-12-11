import argparse
import re
from pathlib import Path

import numpy as np
import soundfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, default="combined")
    parser.add_argument("annotation_file", type=Path, default="annot.txt")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError("Directory doesn't exist: " + args.input_dir)

    annot_file = open(args.annotation_file, 'w')

    recording_dirs = filter(Path.is_dir, args.input_dir.glob('*'))
    for recording_dir in sorted(recording_dirs):
        if not recording_dir.is_dir():
            continue

        operator_audio, _ = soundfile.read(
            recording_dir / 'operator_audio.wav')
        user_audio, _ = soundfile.read(recording_dir / 'user_audio.wav')
        with open(recording_dir / 'emotions.txt') as fid:
            lines = [l.strip() for l in fid][1:]  # Ignore header line
            emotion_data = np.array([
                [float(x) for x in l.split()] for l in lines])

        user_turns = {}
        operator_turns = {}
        for d, f in [(user_turns, 'words_user.txt'),
                     (operator_turns, 'words_operator.txt')]:
            with open(recording_dir / f) as fid:
                turn = 0
                for line in fid:
                    if line.startswith('---'):
                        m = re.search(r'---recording.*turn ([0-9]+)---', line)
                        turn = int(m.group(1))
                    else:
                        m = re.search(r'([0-9]+) ([0-9]+) <?([A-Z\'?!]+)>?',
                                      line)
                        if m:
                            start = int(m.group(1)) * 16
                            end = int(m.group(2)) * 16
                            word = m.group(3)
                            if turn not in d:
                                d[turn] = []
                            d[turn].append((start, end, word))

        for d, p, audio in [(user_turns, 'u', user_audio),
                            (operator_turns, 'o', operator_audio)]:
            out_dir = recording_dir / 'turns'
            out_dir.mkdir(exist_ok=True)
            for turn, words in sorted(d.items()):
                start = words[0][0]
                end = words[-1][1]

                name = "{:02d}_{}_{:03d}".format(int(recording_dir.stem), p,
                                                 turn)
                filename = name + '.wav'
                soundfile.write(out_dir / filename, audio[start:end],
                                samplerate=16000)

                if p == 'u':
                    start_idx, end_idx = np.searchsorted(
                        emotion_data[:, 0],
                        [start / 16000, end / 16000]
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
