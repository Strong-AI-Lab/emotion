"""Filters predictions in a stratified manner."""

from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from emorec.dataset import corpora
from emorec.utils import PathlibPath


@click.command()
@click.argument('input', type=PathlibPath(exists=True, dir_okay=False))
@click.argument('output', type=PathlibPath(dir_okay=False))
@click.option('--corpus', required=True)
def main(input: Path, output: Path, corpus: str):
    df = pd.read_csv(input, header=0)
    df = df.sort_values('Score', ascending=False)
    df['Speaker'] = df['Clip'].map(corpora[corpus.lower()].get_speaker)

    print("Scores for {} clips.".format(len(df)))
    print(df['Score'].describe())
    print()

    hist, bins = np.histogram(df['Score'], bins=200, range=(0, 1))
    bins += 1 / 400
    plt.bar(bins[:-1], hist, width=1 / 200)
    plt.vlines([df['Score'].mean(), df['Score'].median()], ymin=0,
               ymax=hist.max(), colors=['red', 'orange'])
    plt.title("Histogram of scores")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.show()

    sp_cls = df.groupby(['Speaker']).size().sort_values()
    print("Speaker counts:")
    print(sp_cls)
    print()

    counts = defaultdict(int)
    for _, row in df.iterrows():
        counts[row['Speaker']] += 1
        if len([x for x in counts if counts[x] >= 20]) >= 5:
            break

    top_sp = [x for x in counts if counts[x] >= 20]
    print("Top scoring speakers")
    print(df.head(sum(counts.values()))['Speaker'].value_counts()[top_sp])
    print()

    groups = df[df['Speaker'].isin(top_sp)].groupby('Speaker')
    top_speakers = pd.concat([groups.head(20), groups.tail(20)])
    top_speakers = top_speakers.sort_values(['Speaker', 'Score'])

    output.parent.mkdir(parents=True, exist_ok=True)
    top_speakers[['Clip', 'Score']].to_csv(output, header=True, index=False)
    print("Wrote CSV to {}".format(output))


if __name__ == '__main__':
    main()
