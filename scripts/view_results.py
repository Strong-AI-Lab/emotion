#!/usr/bin/python3

import argparse
import re
from itertools import product
from math import isnan
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SUBSTITUTIONS = {
    'cnn/aldeneh': 'CNN',
    'dnn/basic': 'Basic MLP',
    'dnn/1layer': 'FCN (1 layer)',
    'dnn/2layer': 'FCN (2 layers)',
    'dnn/3layer': 'FCN (3 layers)',
    'svm/linear': 'SVM (Linear)',
    'svm/poly2': 'SVM (Quadratic)',
    'svm/poly3': 'SVM (Cubic)',
    'svm/rbf': 'SVM (RBF)',

    'logmel': 'log MFB',
    'eGeMAPS': 'eGeMAPS',
    'GeMAPS': 'GeMAPS',
    'IS09_emotion': 'IS09',
    'IS13_ComParE': 'IS13',
    'audeep': 'auDeep',
    'boaw_eGeMAPS_20_500': 'BoAW: eGeMAPS (20, 500)',
    'boaw_eGeMAPS_50_1000': 'BoAW: eGeMAPS (50, 1000)',
    'boaw_eGeMAPS_100_5000': 'BoAW: eGeMAPS (100, 5000)',
    'boaw_mfcc_le_20_500': 'BoAW: MFCC (20, 500)',
    'boaw_mfcc_le_50_1000': 'BoAW: MFCC (50, 1000)',
    'boaw_mfcc_le_100_5000': 'BoAW: MFCC (100, 5000)',

    'cafe': 'CaFE',
    'crema-d': 'CREMA-D',
    'demos': 'DEMoS',
    'emodb': 'EMO-DB',
    'emofilm': 'EmoFilm',
    'enterface': 'eNTERFACE',
    'iemocap': 'IEMOCAP',
    'jl': 'JL Corpus',
    'msp-improv': 'MSP-IMPROV',
    'portuguese': 'Portuguese',
    'ravdess': 'RAVDESS',
    'savee': 'SAVEE',
    'shemo': 'ShEMO',
    'smartkom': 'SmartKom',
    'tess': 'TESS'
}

feat_col_order = [
    'IS09',
    'IS13',
    'GeMAPS',
    'eGeMAPS',
    'auDeep',
    'BoAW: eGeMAPS (20, 500)',
    'BoAW: eGeMAPS (50, 1000)',
    'BoAW: eGeMAPS (100, 5000)',
    'BoAW: MFCC (20, 500)',
    'BoAW: MFCC (50, 1000)',
    'BoAW: MFCC (100, 5000)',
    'log MFB'
]

feat_cols_subset = [
    'IS09',
    'IS13',
    'GeMAPS',
    'eGeMAPS',
    'auDeep',
    'BoAW: MFCC (20, 500)',
    'BoAW: MFCC (50, 1000)',
    'BoAW: MFCC (100, 5000)',
    'log MFB'
]

clf_col_order = [
    'SVM (Linear)',
    'SVM (Quadratic)',
    'SVM (Cubic)',
    'SVM (RBF)',
    'Basic MLP',
    'FCN (1 layer)',
    'FCN (2 layers)',
    'FCN (3 layers)',
    'CNN'
]

parser = argparse.ArgumentParser()
parser.add_argument('--results', nargs='+', type=Path, required=True)
parser.add_argument('-m', '--metrics', nargs='*', type=str,
                    help="Metrics to output.")
parser.add_argument('--plot', help="Show matplotlib figures",
                    action='store_true')
parser.add_argument('-o', '--output', type=Path,
                    help="Directory to write output summary tables.")
parser.add_argument('-r', '--regex', type=str, default='.*')


def fmt(f):
    if not isnan(f):
        return '{:0.3f}'.format(f)
    return ''


def to_latex(df: pd.DataFrame, output: str, label: str, caption: str,
             bold: str = None):
    df = df.copy()  # So that we don't modify the original frame
    df.columns = (
        df.columns.map(
            lambda x: '\\rotatebox{{90}}{{{}}}'.format(x).replace('_', '\\_'))
        .map(lambda x: SUBSTITUTIONS.get(x, x))
    )
    df.columns.name = None
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.map(
            lambda t: tuple([x.replace('_', '\\_') for x in t]))
    else:
        df.index = (
            df.index.map(lambda x: x.replace('_', '\\_'))
                    .map(lambda x: SUBSTITUTIONS.get(x, x))
        )

    df.to_latex(output, float_format=fmt, longtable=False, na_rep='',
                escape=False, caption=caption, label=label)


def plot_matrix(df, size=(4, 4), rect=[0.25, 0.25, 0.75, 0.75]):
    fig: plt.Figure = plt.figure(figsize=size)
    ax: plt.Axes = fig.add_axes(rect)
    arr = df.to_numpy()
    im = ax.imshow(arr, interpolation='nearest', aspect=0.5)
    im.set_clim(0, 1)

    xlabels = list(df.columns)
    ylabels = list(df.index)

    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    thresh = (arr.max() + arr.min()) / 2.0
    for i, j in product(range(len(ylabels)), range(len(xlabels))):
        color = cmap_max if arr[i, j] < thresh else cmap_min
        ax.text(j, i, format(arr[i, j], '.2f'), ha="center", va="center",
                color=color, fontweight='normal', fontsize='small')

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha='right',
                       rotation_mode='anchor')
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)

    return fig


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
        # Pool reps and take the mean across classes
        uar = {c: df_list[c][('uar', 'all')].stack().mean(1) for c in df_list}

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
    df.columns.names = [None, 'metric']

    metrics = ['mean']
    if args.metrics:
        metrics += args.metrics
    if len(metrics) == 1:
        metrics = metrics[0]
    df = df.xs(metrics, axis=1, level='metric')

    df.columns = df.columns.map(lambda x: SUBSTITUTIONS.get(x, x))
    df.index = df.index.map(lambda t: tuple([SUBSTITUTIONS.get(x, x)
                                             for x in t]))

    best = pd.concat([df.idxmax(0), df.max(0)], 1)
    print(best)

    mean_clf = df.mean(level=0).T
    mean_feat = df.mean(level=1).T
    max_clf = df.max(level=0).T
    max_feat = df.max(level=1).T

    mean_feat = mean_feat[feat_col_order]
    max_feat = max_feat[feat_col_order]
    mean_clf = mean_clf[clf_col_order]
    max_clf = max_clf[clf_col_order]

    DIR = Path('~/Documents/Windows/Uni Stuff/PhD/papers/comparative/images').expanduser()  # noqa: E501
    with PdfPages(DIR / 'mean_clf_matrix.pdf') as pdf:
        fig = plot_matrix(mean_clf, size=(4.5, 4))
        pdf.savefig(fig)

    with PdfPages(DIR / 'max_clf_matrix.pdf') as pdf:
        fig = plot_matrix(max_clf, size=(4.5, 4))
        pdf.savefig(fig)

    with PdfPages(DIR / 'mean_feat_matrix.pdf') as pdf:
        fig = plot_matrix(mean_feat, size=(5.5, 4.5),
                          rect=[0.2, 0.35, 0.8, 0.65])
        pdf.savefig(fig)

    with PdfPages(DIR / 'max_feat_matrix.pdf') as pdf:
        fig = plot_matrix(max_feat[feat_cols_subset], size=(4.5, 4.5),
                          rect=[0.25, 0.3, 0.75, 0.7])
        pdf.savefig(fig)
    plt.show()

    print("Data table:")
    print(df.to_string(float_format=fmt))
    print()
    print(df.swaplevel().sort_index().to_string(float_format=fmt))
    print()
    print("Mean average UAR for each classifier:")
    print(mean_clf.to_string(float_format=fmt))
    print()
    print("Mean average UAR for each feature set:")
    print(mean_feat.to_string(float_format=fmt))
    print()
    print("Best average UAR achieved for each classifier:")
    print(max_clf.to_string(float_format=fmt))
    print()
    print("Best average UAR achieved for each feature set:")
    print(max_feat.to_string(float_format=fmt))

    if args.output:
        to_latex(df, args.output / 'combined.tex',
                 label='tab:CombinedResults', caption="Combined results table")
        to_latex(
            mean_clf, args.output / 'mean_clf.tex',
            label='tab:MeanClassifier',
            caption="Mean average UAR for each classifier.",
        )
        to_latex(
            mean_feat, args.output / 'mean_feat.tex',
            label='tab:MeanFeature',
            caption="Mean average UAR for each feature set."
        )
        to_latex(
            max_clf, args.output / 'max_clf.tex',
            label='tab:MaxClassifier',
            caption="Maximum average UAR for each classifier."
        )
        to_latex(
            max_feat[feat_cols_subset], args.output / 'max_feat.tex',
            label='tab:MaxFeature',
            caption="Maximum average UAR for each feature set."
        )
    if args.plot:
        plt.show()


if __name__ == "__main__":
    main()
