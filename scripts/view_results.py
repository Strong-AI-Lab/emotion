#!/usr/bin/python3

import argparse
import os
from glob import glob

import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('directory')


def main():
    args = parser.parse_args()

    # figures = []
    for d in glob(os.path.join(args.directory, '*')):
        if not os.path.isdir(d):
            continue

        df_list = {}
        for filename in sorted(glob(os.path.join(d, '*.csv'))):
            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            df_list[name] = pd.read_csv(filename, header=[0, 1, 2, 3],
                                        index_col=0)
        uar = {c: df_list[c][('uar', 'all')].mean(1) for c in df_list}

        fig, ax = plt.subplots()
        fig: plt.Figure
        ax: plt.Axes

        ax.set_title(os.path.basename(d))
        ax.boxplot(uar.values(), labels=uar.keys(), notch=True, bootstrap=1000)
        ax.set_xticklabels(uar.keys(), rotation=45)
        ax.set_xlabel('Config')
        ax.set_ylabel('UAR')
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
