#!/usr/bin/python3

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs='+')
parser.add_argument('--show', help="Show matplotlib figures",
                    action='store_true')
parser.add_argument('-o', '--output', type=str,
                    help="Directory to write output summary tables.")


def main():
    args = parser.parse_args()

    dirs = sorted([Path(d) for d in args.directory])
    uar_list = []
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

        df_uar = pd.Series(list(table.values()), index=table.keys())
        uar_list.append(df_uar)
    df = pd.concat(uar_list, keys=[d.name for d in dirs],
                   names=['corpus', 'classifier', 'config']).unstack(0)
    print("Data table:")
    print(df)
    print()
    print("Mean across classifiers:")
    print(df.mean(level=0))
    print()
    print("Mean across configs:")
    print(df.mean(level=1))
    print()
    print("Max across configs:")
    print(df.max(level=0))
    print()
    print("Max across configs:")
    print(df.max(level=1))

    if args.output:
        df.to_latex(Path(args.output) / 'combined.tex')
        df.mean(level=0).to_latex(Path(args.output) / 'mean_clf.tex')
        df.mean(level=1).to_latex(Path(args.output) / 'mean_config.tex')
        df.max(level=0).to_latex(Path(args.output) / 'max_clf.tex')
        df.max(level=1).to_latex(Path(args.output) / 'max_config.tex')
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
