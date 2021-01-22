import argparse
import itertools
import logging
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from emotion_recognition.utils import itmap, ordered_intersect
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.stats.anova import AnovaRM

SUBSTITUTIONS = {
    # Classifiers
    'aldeneh2017': 'Aldeneh (2017)',
    'latif2019': 'Latif (2019)',
    'zhang2019': 'Zhang (2019)',
    'zhao2019': 'Zhao (2019)',
    'mlp/1layer': 'FCN (1 layer)',
    'mlp/2layer': 'FCN (2 layers)',
    'mlp/3layer': 'FCN (3 layers)',
    'svm/linear': 'SVM (Linear)',
    'svm/poly2': 'SVM (Quadratic)',
    'svm/poly3': 'SVM (Cubic)',
    'svm/rbf': 'SVM (RBF)',

    # Feature sets
    'logmel_40': '40 vlen log-mspec',
    'logmel_240': '240 vlen log-mspec',
    'spectrograms_40': '5s x 40 clip log-mspec.',
    'spectrograms_240': '5s x 240 clip log-mspec.',
    'raw_audio': 'Raw audio',
    'eGeMAPS': 'eGeMAPS',
    'GeMAPS': 'GeMAPS',
    'IS09': 'IS09',
    'IS13': 'IS13',
    'audeep-0.05-0.025-240-60_b64_l0.001': 'auDeep',
    'boaw_20_500': 'BoAW (20, 500)',
    'boaw_50_1000': 'BoAW (50, 1000)',
    'boaw_100_5000': 'BoAW (100, 5000)',
    'bow': 'Bag of words',

    # Dataset initials
    'cafe': 'CA',
    'crema-d': 'CR',
    'demos': 'DE',
    'emodb': 'ED',
    'emofilm': 'EF',
    'enterface': 'EN',
    'iemocap': 'IE',
    'jl': 'JL',
    'msp-improv': 'MS',
    'portuguese': 'PO',
    'ravdess': 'RA',
    'savee': 'SA',
    'shemo': 'SH',
    'smartkom': 'SM',
    'tess': 'TE'
}


def subs(x: str) -> str:
    return SUBSTITUTIONS.get(x, x)


def latex_subs(x: str) -> str:
    return x.replace('_', '\\_')


def substitute_labels(df: pd.DataFrame):
    df.index = df.index.map(itmap(subs))
    df.columns = df.columns.map(itmap(subs))


def fmt(f: float) -> str:
    if not math.isnan(f):
        return '{:#.3g}'.format(100 * f)
    return ''


def to_latex(df: pd.DataFrame, output: str, label: str, caption: str):
    df = df.copy()  # So that we don't modify the original frame
    df.columns = df.columns.map(
        itmap(lambda x: '\\rotatebox{{90}}{{{}}}'.format(x)))
    df.columns = df.columns.map(itmap(latex_subs)).map(itmap(subs))
    df.columns.name = None
    df.index = df.index.map(itmap(latex_subs)).map(itmap(subs))
    df.to_latex(output, float_format=fmt, longtable=False, na_rep='',
                escape=False, caption=caption, label=label)


def plot_matrix(df: pd.DataFrame, size: tuple = (4, 4),
                rect: list = [0.25, 0.25, 0.75, 0.75]) -> plt.Figure:
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(rect)
    arr = df.to_numpy()
    im = ax.imshow(arr, interpolation='nearest', aspect=0.5)
    im.set_clim(0, 1)

    xlabels = list(df.columns)
    ylabels = list(df.index)

    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    thresh = 0.5
    for i, j in itertools.product(range(len(ylabels)), range(len(xlabels))):
        color = cmap_max if arr[i, j] < thresh else cmap_min
        ax.text(j, i, fmt(arr[i, j]), ha="center", va="center",
                color=color, fontweight='normal', fontsize='small')

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha='right', fontsize='small',
                       rotation_mode='anchor')
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize='small')

    return fig


def main():
    logging.basicConfig()
    logger = logging.getLogger('view_results')
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--metrics', nargs='*', type=str,
                        help="Metrics to output.")
    parser.add_argument('--plot', action='store_true',
                        help="Show matplotlib figures.")
    parser.add_argument('--latex', type=Path,
                        help="Directory to output latex tables.")
    parser.add_argument('--regex', type=str, default='.*')
    parser.add_argument('--images', type=Path,
                        help="Directory to output images.")
    parser.add_argument('--excel', type=Path,
                        help="Directory to output Excel files.")
    parser.add_argument('--raw', action='store_true',
                        help="Don't rename and reorder columns.")
    parser.add_argument(
        '--anova', action='store_true', help="Perform repeated measures ANOVA "
        "on classifiers x features for all datasets."
    )
    args = parser.parse_args()

    if args.results.is_file():
        logger.info("Found results cache at {}.".format(str(args.results)))
        df = pd.read_csv(args.results, index_col=[0, 1, 2])
    else:
        dirs = sorted([d for d in args.results.glob('*') if d.is_dir()])

        uar_list = {}
        for d in dirs:
            logger.info("Reading directory {}".format(d))
            table = {}
            df_list = {}
            for filename in [x for x in sorted(d.glob('**/*.csv'))
                             if re.search(args.regex, str(x))]:
                logger.debug("Found file {}".format(filename))
                name = filename.relative_to(d).with_suffix('')
                df_list[name] = pd.read_csv(filename, header=0, index_col=0)
            if not df_list:
                continue

            for name, df in df_list.items():
                p = name.parts
                # name is assumed to be cla/ssi/fier/features
                table[('/'.join(p[:-1]), p[-1])] = (
                    df['uar'].mean(), df['uar'].std(), df['uar'].max())

            # multi-index, each value in table is a tuple of 3 floats
            df_uar = pd.DataFrame(table.values(), index=table.keys(),
                                  columns=['mean', 'std', 'max'])
            # d.name is assumed to be the corpus name
            uar_list[d.name] = df_uar

        if not uar_list:
            raise FileNotFoundError("No valid files found matching regex.")

        # Concatenate DataFrames for all datasets
        df = pd.concat(uar_list, names=['Dataset', 'Classifier', 'Features'])
        df.to_csv('/tmp/emotion_results.csv')

    # Move corpus to columns, outermost level
    df = df.unstack(0).swaplevel(axis=1)
    df.columns.names = ['Corpus', 'Metric']

    metrics = {'mean'}
    if args.metrics:
        metrics = metrics.union(args.metrics)
    metrics = tuple(metrics)
    if len(metrics) == 1:
        metrics = metrics[0]

    df = df.loc[:, (slice(None), metrics)]
    df = df.sort_index(axis=1, level=0)

    if not args.raw:
        df.columns = df.columns.map(itmap(subs))
        df.index = df.index.map(itmap(subs))

    full = df
    df = df.loc[:, (slice(None), 'mean')].droplevel('Metric', axis=1)

    # Get the best (classifier, features) combination and corresponding UAR
    best = pd.concat([df.idxmax(0), df.max(0)], 1, keys=['Combination', 'UAR'])
    best['Classifier'] = best['Combination'].map(lambda t: t[0])
    best['Features'] = best['Combination'].map(lambda t: t[1])
    best.drop(columns='Combination')
    best = best[['Classifier', 'Features', 'UAR']]
    substitute_labels(best)

    # Classifier-features table
    clf_feat = df.mean(1).unstack(1)

    # {mean, max} per {classifier, features}
    mean_clf = df.mean(level=0).T
    mean_feat = df.mean(level=1).T
    max_clf = df.max(level=0).T
    max_feat = df.max(level=1).T

    if not args.raw:
        # Order the columns how we want
        clf_feat = clf_feat.loc[
            ordered_intersect(SUBSTITUTIONS, clf_feat.index),
            ordered_intersect(SUBSTITUTIONS, clf_feat.columns)
        ]
        substitute_labels(clf_feat)
        mean_feat = mean_feat[ordered_intersect(SUBSTITUTIONS,
                                                mean_feat.columns)]
        substitute_labels(mean_feat)
        max_feat = max_feat[ordered_intersect(SUBSTITUTIONS,
                                              max_feat.columns)]
        substitute_labels(max_feat)
        mean_clf = mean_clf[ordered_intersect(SUBSTITUTIONS, mean_clf.columns)]
        substitute_labels(mean_clf)
        max_clf = max_clf[ordered_intersect(SUBSTITUTIONS, max_clf.columns)]
        substitute_labels(max_clf)

    print("Data table:")
    print(full.to_string(float_format=fmt))
    print()
    print(full.swaplevel().sort_index().to_string(float_format=fmt))
    print()

    if args.anova:
        flat = df.stack().reset_index().rename(columns={0: 'UAR'})
        flat = flat[~flat['Classifier'].isin(['aldeneh2017', 'latif2019',
                                              'zhang2019', 'zhao2019'])]
        if not flat.empty:
            model = AnovaRM(flat, 'UAR', 'Corpus', within=['Classifier',
                                                           'Features'])
            res = model.fit()
            print(res.anova_table)

    print("Best classifier-features combinations:")
    print(best.to_string(float_format=fmt))
    print()

    print("Average UAR of (classifier, feature) pairs:")
    print(clf_feat.to_string(float_format=fmt))
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

    if args.images or args.plot:
        mean_clf_fig = plot_matrix(mean_clf, size=(4, 3.5))
        max_clf_fig = plot_matrix(max_clf, size=(4, 3.5))
        mean_feat_fig = plot_matrix(mean_feat, size=(5, 3.75),
                                    rect=[0.2, 0.35, 0.8, 0.65])
        max_feat_fig = plot_matrix(max_feat, size=(4, 3.5),
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

        if args.plot:
            plt.show()

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
            max_feat, args.latex / 'max_feat.tex',
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
            [full.stack(), clf_feat, mean_clf, mean_feat, max_clf, max_feat],
            ['combined', 'clf_feat', 'mean_clf', 'mean_feat', 'max_clf',
             'max_feat']
        ):
            tab.to_excel(args.excel / (name + '.xlsx'), merge_cells=False)


if __name__ == "__main__":
    main()
