#!/usr/bin/python3

import argparse
import re
from math import isnan
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results', nargs='+', type=Path, required=True)
parser.add_argument('-m', '--metrics', nargs='*', type=str,
                    help="Metrics to output.")
parser.add_argument('--plot', help="Show matplotlib figures",
                    action='store_true')
parser.add_argument('-o', '--output', type=Path,
                    help="Directory to write output summary tables.")
parser.add_argument('-r', '--regex', type=str, default='.*')


def main():
    args = parser.parse_args()

    _dirs = [d for d in args.results if d.is_dir()]
    dirs = []
    for d in _dirs:
        if '*' in d.stem:
            dirs.extend(d.parent.glob(d.stem))
        else:
            dirs.append(d)
    dirs = sorted(dirs)

    uar_list = {}
    for d in dirs:
        print("Reading", d)
        table = {}
        df_list = {}
        for filename in [x for x in sorted(d.glob('**/*.csv'))
                         if re.search(args.regex, str(x))]:
            name = filename.relative_to(d).with_suffix('')
            df_list[name] = pd.read_csv(filename, header=[0, 1, 2, 3],
                                        index_col=0)
        if not df_list:
            continue
        uar = {c: df_list[c][('uar', 'all')].mean(1) for c in df_list}

        for name in df_list:
            config = Path(name).name
            clf = str(Path(name).parent)
            table[(clf, config)] = (uar[name].mean(), uar[name].std(),
                                    uar[name].max())

        if args.plot:
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.add_subplot(1, 1, 1)
            ax.set_title(d.name)
            ax.boxplot(uar.values(), labels=uar.keys(), notch=True,
                       bootstrap=1000)
            ax.set_xticklabels(uar.keys(), rotation=90)
            ax.set_xlabel('Config')
            ax.set_ylabel('UAR')
            fig.tight_layout()

        df_uar = pd.DataFrame(list(table.values()), index=table.keys(),
                              columns=['mean', 'std', 'max'])
        uar_list[d.name] = df_uar

    if not uar_list:
        raise FileNotFoundError("No valid files found matching regex.")

    df = pd.concat(
        uar_list.values(), keys=[name for name in uar_list],
        names=['corpus', 'classifier', 'config']
    ).unstack(0).swaplevel(axis=1)

    def fmt(f):
        if not isnan(f):
            return '{:0.3f}'.format(f)
        return ''
    FMT = fmt

    metrics = ['mean']
    if args.metrics:
        metrics += args.metrics
    df = df.loc[:, (slice(None), metrics)]

    print("Data table:")
    print(df.to_string(float_format=FMT))
    print()
    print(df.swaplevel().sort_index().to_string(float_format=FMT))
    print()
    print("Mean average UAR across configs and corpora:")
    print(df.mean(level=0).to_string(float_format=FMT))
    print()
    print("Mean average UAR across classifiers and corpora:")
    print(df.mean(level=1).to_string(float_format=FMT))
    print()
    print("Maximum average UAR across configs and corpora:")
    print(df.max(level=0).to_string(float_format=FMT))
    print()
    print("Maximum average UAR across classifiers and corpora:")
    print(df.max(level=1).to_string(float_format=FMT))

    if args.output:
        common = dict(float_format=FMT, longtable=True, na_rep='')
        df.to_latex(args.output / 'combined.tex', **common,
                    caption="Combined results table")
        df.mean(level=0).to_latex(
            args.output / 'mean_config.tex', **common,
            caption="Mean average UAR across configs and corpora"
        )
        df.mean(level=1).to_latex(
            args.output / 'mean_clf.tex', **common,
            caption="Mean average UAR across classifiers and corpora"
        )
        df.max(level=0).to_latex(
            args.output / 'max_config.tex', **common,
            caption="Maximum average UAR across configs and corpora"
        )
        df.max(level=1).to_latex(
            args.output / 'max_clf.tex', **common,
            caption="Maximum average UAR across classifiers and corpora"
        )
    if args.plot:
        plt.show()


if __name__ == "__main__":
    main()
