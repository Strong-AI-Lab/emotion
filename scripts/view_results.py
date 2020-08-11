#!/usr/bin/python3

import argparse
import logging
import re
from itertools import product, combinations
from math import isnan
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_rel

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
    'audeep_all': 'auDeep (all datasets)',
    'boaw_eGeMAPS_20_500': 'BoAW: eGeMAPS (20, 500)',
    'boaw_eGeMAPS_50_1000': 'BoAW: eGeMAPS (50, 1000)',
    'boaw_eGeMAPS_100_5000': 'BoAW: eGeMAPS (100, 5000)',
    'boaw_mfcc_le_20_500': 'BoAW: MFCC (20, 500)',
    'boaw_mfcc_le_50_1000': 'BoAW: MFCC (50, 1000)',
    'boaw_mfcc_le_100_5000': 'BoAW: MFCC (100, 5000)',
    'bow': 'Bag of words',

    'cafe': 'CaFE',
    'crema-d_nominal': 'CREMA-D Nom.',
    'crema-d_multimodal': 'CREMA-D AV',
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
    'auDeep',
    'auDeep (all datasets)',
    'IS09',
    'IS13',
    'GeMAPS',
    'eGeMAPS',
    'BoAW: eGeMAPS (20, 500)',
    'BoAW: eGeMAPS (50, 1000)',
    'BoAW: eGeMAPS (100, 5000)',
    'BoAW: MFCC (20, 500)',
    'BoAW: MFCC (50, 1000)',
    'BoAW: MFCC (100, 5000)',
    'log MFB'
]

feat_cols_subset = [
    'auDeep',
    'auDeep (all datasets)',
    'IS09',
    'IS13',
    'GeMAPS',
    'eGeMAPS',
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


def fmt(f):
    if not isnan(f):
        return '{:#.3g}'.format(100 * f)
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

    thresh = 0.5
    for i, j in product(range(len(ylabels)), range(len(xlabels))):
        color = cmap_max if arr[i, j] < thresh else cmap_min
        ax.text(j, i, fmt(arr[i, j]), ha="center", va="center",
                color=color, fontweight='normal', fontsize='small')

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha='right', fontsize='small',
                       rotation_mode='anchor')
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize='small')

    return fig


def ordered_intersect(a, b):
    return [x for x in a if x in b]


def main():
    logging.basicConfig()
    logger = logging.getLogger('view_results')
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+', type=Path, required=True)
    parser.add_argument('--metrics', nargs='*', type=str,
                        help="Metrics to output.")
    parser.add_argument('--plot', help="Show matplotlib figures.",
                        action='store_true')
    parser.add_argument('--print', help="Print tables to console.",
                        action='store_true')
    parser.add_argument('--latex', type=Path,
                        help="Directory to output latex tables.")
    parser.add_argument('--regex', type=str, default='.*')
    parser.add_argument('--images', type=Path,
                        help="Directory to output images.")
    parser.add_argument('--excel', type=Path,
                        help="Directory to output Excel files.")
    args = parser.parse_args()

    cache_file = Path(args.results[0])
    if len(args.results) == 1 and cache_file.is_file():
        logger.info("Found results cache at {}.".format(str(cache_file)))
        df = pd.read_csv(cache_file, index_col=[0, 1, 2])
    else:
        _dirs = [d for d in args.results if d.is_dir()]
        dirs = []
        for d in _dirs:
            # This takes into account Windows CMD globbing issues
            if '*' in d.stem:
                dirs.extend(d.parent.glob(d.stem))
            else:
                dirs.append(d)
        dirs = sorted(dirs)

        uar_list = {}
        for d in dirs:
            logger.info("Reading directory {}".format(d))
            table = {}
            df_list = {}
            for filename in [x for x in sorted(d.glob('**/*.csv'))
                             if re.search(args.regex, str(x))]:
                name = filename.relative_to(d).with_suffix('')
                logger.debug("Found file {}".format(str(filename)))
                df_list[name] = pd.read_csv(filename, header=[0, 1, 2, 3],
                                            index_col=0)
            if not df_list:
                continue
            # Pool reps and take the mean across classes
            uar = {c: df_list[c][('uar', 'all')].stack().mean(1)
                   for c in df_list}

            for name in df_list:
                feat = Path(name).name
                clf = str(Path(name).parent)
                table[(clf, feat)] = (uar[name].mean(), uar[name].std(),
                                      uar[name].max())

            df_uar = pd.DataFrame(list(table.values()), index=table.keys(),
                                  columns=['mean', 'std', 'max'])
            uar_list[d.name] = df_uar

        if not uar_list:
            raise FileNotFoundError("No valid files found matching regex.")

        df = pd.concat(uar_list.values(), keys=uar_list.keys(),
                       names=['Dataset', 'Classifier', 'Config'])
        df.to_csv('/tmp/emotion_results.csv')

    df = df.unstack(0).swaplevel(axis=1)
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

    best = pd.concat([df.idxmax(0), df.max(0)], 1, keys=['Combination', 'UAR'])
    best['Classifier'] = best['Combination'].map(lambda t: t[0])
    best['Features'] = best['Combination'].map(lambda t: t[1])
    best.drop(columns='Combination')
    best = best[['Classifier', 'Features', 'UAR']]

    clf_feat = df.mean(1).unstack(1)
    clf_feat = clf_feat.loc[
        ordered_intersect(clf_col_order, clf_feat.index),
        ordered_intersect(feat_cols_subset, clf_feat.columns)
    ]

    mean_clf = df.mean(level=0).T
    mean_feat = df.mean(level=1).T
    max_clf = df.max(level=0).T
    max_feat = df.max(level=1).T

    logging.debug(ordered_intersect(feat_col_order, mean_feat.columns))
    mean_feat = mean_feat[ordered_intersect(feat_col_order, mean_feat.columns)]
    max_feat = max_feat[ordered_intersect(feat_col_order, max_feat.columns)]
    max_feat_subset = max_feat[
        ordered_intersect(feat_cols_subset, max_feat.columns)]
    mean_clf = mean_clf[ordered_intersect(clf_col_order, mean_clf.columns)]
    max_clf = max_clf[ordered_intersect(clf_col_order, max_clf.columns)]

    diff_feat = max_feat - mean_feat
    diff_clf = max_clf - mean_clf

    if args.print:
        print("Data table:")
        print(df.to_string(float_format=fmt))
        print()
        print(df.swaplevel().sort_index().to_string(float_format=fmt))
        print()

        print("Best classifier-features combinations:")
        print(best)
        print()

        print("Average UAR of (classifier, feature) pairs:")
        print(clf_feat)
        print()

        print("T-tests on columns:")
        for c1, c2 in combinations(clf_feat.columns, 2):
            tstat, pval = ttest_rel(clf_feat.loc[:, c1], clf_feat.loc[:, c2],
                                    nan_policy='omit')
            if pval < 0.05 / 28:
                print('{}, {}, {:.3g}, {:.3g}'.format(c1, c2, tstat, pval))
        print()
        print("T-tests on rows:")
        for r1, r2 in combinations(clf_feat.index, 2):
            tstat, pval = ttest_rel(clf_feat.loc[r1, :], clf_feat.loc[r2, :],
                                    nan_policy='omit')
            if pval < 0.05 / 28:
                print('{}, {}, {:.3g}, {:.3g}'.format(r1, r2, tstat, pval))
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

        print("Increase in features UAR:")
        print(diff_feat)
        print()
        print("Increase in classifiers UAR:")
        print(diff_clf)
        print()

    if args.images or args.plot:
        mean_clf_fig = plot_matrix(mean_clf, size=(4, 3.5))
        max_clf_fig = plot_matrix(max_clf, size=(4, 3.5))
        mean_feat_fig = plot_matrix(mean_feat, size=(5, 3.75),
                                    rect=[0.2, 0.35, 0.8, 0.65])
        max_feat_fig = plot_matrix(max_feat_subset, size=(4, 3.5),
                                   rect=[0.25, 0.3, 0.75, 0.7])

        if args.images:
            with PdfPages(args.images / 'mean_clf_matrix.pdf') as pdf:
                pdf.savefig(mean_clf_fig)

            with PdfPages(args.images / 'max_clf_matrix.pdf') as pdf:
                pdf.savefig(max_clf_fig)

            with PdfPages(args.images / 'mean_feat_matrix.pdf') as pdf:
                pdf.savefig(mean_feat_fig)

            with PdfPages(args.images / 'max_feat_matrix.pdf') as pdf:
                pdf.savefig(max_feat_fig)

    if args.latex:
        to_latex(df, args.latex / 'combined.tex',
                 label='tab:CombinedResults', caption="Combined results table")
        to_latex(
            mean_clf, args.latex / 'mean_clf.tex',
            label='tab:MeanClassifier',
            caption="Mean average UAR for each classifier.",
        )
        to_latex(
            mean_feat, args.latex / 'mean_feat.tex',
            label='tab:MeanFeature',
            caption="Mean average UAR for each feature set."
        )
        to_latex(
            max_clf, args.latex / 'max_clf.tex',
            label='tab:MaxClassifier',
            caption="Maximum average UAR for each classifier."
        )
        to_latex(
            max_feat_subset, args.latex / 'max_feat.tex',
            label='tab:MaxFeature',
            caption="Maximum average UAR for each feature set."
        )
        to_latex(
            clf_feat, args.latex / 'clf_feat.tex',
            label='tab:FeatClassifier',
            caption="Average performance of (classifier, feature) pairs "
                    "across datasets"
        )

    if args.excel:
        for tab, name in zip(
            [df.stack(), clf_feat, mean_clf, mean_feat, max_clf, max_feat],
            ['combined', 'clf_feat', 'mean_clf', 'mean_feat', 'max_clf',
             'max_feat']
        ):
            tab.to_excel(args.excel / (name + '.xlsx'), merge_cells=False)

    if args.plot:
        plt.show()


if __name__ == "__main__":
    main()
