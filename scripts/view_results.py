#!/usr/bin/python3

import argparse
import os
from math import sqrt, ceil
from glob import glob

import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('directory')


def main():
    args = parser.parse_args()

    dirs = sorted([d for d in glob(os.path.join(args.directory, '*'))
                   if os.path.isdir(d)])
    n_rows = round(sqrt(len(dirs)))
    n_cols = ceil(len(dirs) / n_rows)
    fig: plt.Figure = plt.figure(figsize=(12, 6))
    for idx, d in enumerate(dirs):
        df_list = {}
        for filename in sorted(glob(os.path.join(d, '*.csv'))):
            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            df_list[name] = pd.read_csv(filename, header=[0, 1, 2, 3],
                                        index_col=0)
        uar = {c: df_list[c][('uar', 'all')].mean(1) for c in df_list}

        ax: plt.Axes = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.set_title(os.path.basename(d))
        ax.boxplot(uar.values(), labels=uar.keys(), notch=True, bootstrap=1000)
        ax.set_xticklabels(uar.keys(), rotation=45)
        ax.set_xlabel('Config')
        ax.set_ylabel('UAR')
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
