#!/usr/bin/python3

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs='+')


def main():
    args = parser.parse_args()

    dirs = sorted([Path(d) for d in args.directory])
    for d in dirs:
        table = {}
        df_list = {}
        for filename in sorted(d.glob('**/*.csv')):
            name = filename.relative_to(d).with_suffix('')
            df_list[name] = pd.read_csv(filename, header=[0, 1, 2, 3],
                                        index_col=0)
        if not df_list:
            continue
        uar = {c: df_list[c][('uar', 'all')].mean(1) for c in df_list}

        for name in df_list:
            config = Path(name).name
            clf = str(Path(name).parent)
            table[(clf, config)] = uar[name].mean()

        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot(1, 1, 1)
        ax.set_title(d.name)
        ax.boxplot(uar.values(), labels=uar.keys(), notch=True, bootstrap=1000)
        ax.set_xticklabels(uar.keys(), rotation=90)
        ax.set_xlabel('Config')
        ax.set_ylabel('UAR')
        fig.tight_layout()

        df = pd.DataFrame(table.values(), index=table.keys()).unstack()
        print(df)
    plt.show()


if __name__ == "__main__":
    main()
