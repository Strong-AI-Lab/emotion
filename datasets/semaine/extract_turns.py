import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import soundfile


@click.command()
@click.argument('combined', type=Path, default='combined')
def main(combined: Path):
    """Splits full recordings into individual turns."""
    df = pd.DataFrame(columns=['Activation', 'Expectation', 'Power',
                               'Valence'])
    for recording_dir in filter(Path.is_dir, combined.glob('*')):
        print(recording_dir)

        operator_audio, _ = soundfile.read(recording_dir / 'operator.wav')
        user_audio, _ = soundfile.read(recording_dir / 'user.wav')
        emotions = pd.read_csv(recording_dir / 'emotions.csv', header=0,
                               index_col=0)

        user_turns = {}
        operator_turns = {}
        for d, f in [(user_turns, 'user.txt'),
                     (operator_turns, 'operator.txt')]:
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

                name = f"{int(recording_dir.stem):02d}_{p}_{turn:03d}"
                filename = name + '.wav'
                soundfile.write(out_dir / filename, audio[start:end],
                                samplerate=16000)

                if p == 'u':
                    start_idx, end_idx = np.searchsorted(
                        emotions.index, [start / 16000, end / 16000]
                    )
                    if start_idx != end_idx:
                        mean_emotions = emotions.iloc[
                            start_idx:end_idx, :].mean()
                        df.loc[name, :] = mean_emotions

    df.index.name = 'Name'
    for c in df.columns:
        df[c].to_csv(f'{c.lower()}.csv')


if __name__ == "__main__":
    main()
