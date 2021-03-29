import itertools
import logging
import math
import re
from pathlib import Path
from typing import Dict, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from emorec.utils import PathlibPath, itmap, ordered_intersect
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import friedmanchisquare, rankdata
from statsmodels.stats.anova import AnovaRM


CLF_TIME = {
    # Sequence classifiers
    "aldeneh2017": "ALD",
    "latif2019": "LAT",
    "zhang2019": "ZHN",
    "zhao2019": "ZHO",
}

CLF_GENERIC = {
    "mlp/1layer": "MLP-1",
    "mlp/2layer": "MLP-2",
    "mlp/3layer": "MLP-3",
    "svm/linear": "SVM-L",
    "svm/poly2": "SVM-Q",
    "svm/poly3": "SVM-C",
    "svm/rbf": "SVM-R",
    "rf": "RF",
}

CLF_1D = {
    # 1-D classifiers
    "depinto2020": "DEP",
    "iskhakova2020": "ISK",
}


DATASETS = {
    # Dataset initials
    "cafe": "CA",
    "crema-d": "CR",
    "demos": "DE",
    "emodb": "ED",
    "emofilm": "EF",
    "enterface": "EN",
    "iemocap": "IE",
    "jl": "JL",
    "msp-improv": "MS",
    "portuguese": "PO",
    "ravdess": "RA",
    "savee": "SA",
    "shemo": "SH",
    "smartkom": "SM",
    "tess": "TE",
    "urdu": "UR",
    "venec": "VE",
}

CLASSIFIERS = dict(**CLF_TIME, **CLF_GENERIC, **CLF_1D)
SUBSTITUTIONS = dict(**CLASSIFIERS, **DATASETS)


def subs(x: str) -> str:
    return SUBSTITUTIONS.get(x, x)


def latex_subs(x: str) -> str:
    return subs(x).replace("_", "\\_")


def substitute_labels(df: pd.DataFrame):
    df.index = df.index.map(itmap(subs))
    df.columns = df.columns.map(itmap(subs))


def fmt(f: float) -> str:
    if not math.isnan(f):
        return f"{100 * f:#.3g}"
    return ""


def to_latex(df: pd.DataFrame, output: Path, label: str, caption: str):
    df = df.copy()  # So that we don't modify the original frame
    df.columns = df.columns.map(itmap(lambda x: f"\\rotatebox{{90}}{{{x}}}"))
    df.columns = df.columns.map(itmap(latex_subs))
    df.columns.name = None
    df.index = df.index.map(itmap(latex_subs))
    df.to_latex(
        output,
        float_format=fmt,
        longtable=False,
        na_rep="",
        escape=False,
        caption=caption,
        label=label,
    )


def plot_matrix(df: pd.DataFrame, width: float = 7, cmap: str = "gray") -> plt.Figure:
    arr = df.to_numpy()
    height = int(arr.shape[0] / arr.shape[1] * width)

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    im = ax.imshow(arr, interpolation="nearest", aspect=0.5, cmap=cmap)
    im.set_clim(0, 1)

    xlabels = list(df.columns)
    ylabels = list(df.index)

    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    thresh = 0.5
    for i, j in itertools.product(range(len(ylabels)), range(len(xlabels))):
        color = cmap_max if arr[i, j] < thresh else cmap_min
        ax.text(
            j,
            i,
            fmt(arr[i, j]),
            ha="center",
            va="center",
            color=color,
            fontweight="normal",
            fontsize="small",
        )

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(
        xlabels, rotation=30, ha="right", fontsize="small", rotation_mode="anchor"
    )
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize="small")

    fig.tight_layout()

    return fig


def run_anova(flat: pd.DataFrame):
    grps = set(flat.groupby(["Classifier", "Features", "Corpus"]).groups.keys())
    clf = flat["Classifier"].unique()
    feat = flat["Features"].unique()
    corpora = flat["Corpus"].unique()
    all_grps = set(itertools.product(clf, feat, corpora))
    missing = all_grps - grps
    if len(missing) > 0:
        raise ValueError(f"Missing combinations for ANOVA: {missing}")

    model = AnovaRM(flat, "UAR", "Corpus", within=["Classifier", "Features"])
    res = model.fit()
    print("Repeated-measures ANOVA over features x classifiers:")
    print(res.anova_table)
    print()


def run_friedman(table: pd.DataFrame):
    from Orange.evaluation import graph_ranks
    from statsmodels.stats.libqsturng import qsturng

    _, pvalue = friedmanchisquare(*table.transpose().to_numpy())
    ranktable = rankdata(-table.to_numpy(), axis=1)
    avgrank = ranktable.mean(0)
    k = len(avgrank)
    n = len(ranktable)
    cd = qsturng(0.95, k, np.inf) * np.sqrt((k * (k + 1)) / (12 * n))
    names = list(table.columns)
    df = pd.DataFrame(
        {
            "Mean rank": avgrank,
            "Mean": table.mean(),
            "Std. dev.": table.std(),
            "Median": table.median(),
            "MAD": table.mad(),
        },
        index=names,
    ).sort_values("Mean rank")
    topclf = df.index[0]
    gamma = [0]
    for x in df.index[1:]:
        pmad = np.sqrt(
            ((n - 1) * df.loc[topclf, "MAD"] ** 2 + (n - 1) * df.loc[x, "MAD"] ** 2)
            / (2 * n - 2)
        )
        gamma.append((df.loc[topclf, "Median"] - df.loc[x, "Median"]) / pmad)
    df["Effect size"] = gamma
    print(df.to_string())
    print(f"p = {pvalue}, cd = {cd:.2f} ranks")
    print()
    graph_ranks(avgrank, names, cd, width=4, reverse=True)


@click.command()
@click.argument("results", type=PathlibPath(exists=True))
@click.option(
    "--metrics",
    type=click.Choice(["mean", "std", "max"]),
    multiple=True,
    default=("mean",),
    help="Metrics to output.",
)
@click.option("--plot", is_flag=True, help="Show matplotlib figures.")
@click.option("--latex", type=Path, help="Directory to output latex tables.")
@click.option("--regex", type=str, default=".*", help="Regex to limit results.")
@click.option("--images", type=Path, help="Directory to output images.")
@click.option("--excel", type=Path, help="Directory to output Excel files.")
@click.option("--substitute", is_flag=True, help="Don't rename and reorder columns.")
@click.option(
    "--test",
    type=click.Choice(["anova", "friedman"]),
    help="Perform various statistical tests.",
)
@click.option("--cmap", type=str, help="Colormap to use for plots.")
def main(
    results: Path,
    metrics: Tuple[str],
    plot: bool,
    latex: Path,
    regex: str,
    images: Path,
    excel: Path,
    substitute: bool,
    test: str,
    cmap: str,
):
    """Analyse, view and plot results from directory RESULTS. The
    directory is search recursively for experiment CSVs.
    """

    logging.basicConfig()
    logger = logging.getLogger("view_results")
    logger.setLevel(logging.INFO)

    if results.is_file():
        logger.info(f"Found results cache at {results}")
        df: pd.DataFrame = pd.read_csv(results, index_col=[0, 1, 2])
    else:
        dirs = sorted([d for d in results.glob("*") if d.is_dir()])

        uar_list = {}
        for d in dirs:
            logger.info(f"Reading directory {d}")
            table = {}
            df_list: Dict[Path, pd.DataFrame] = {}
            for filename in [
                x for x in sorted(d.glob("**/*.csv")) if re.search(regex, str(x))
            ]:
                logger.debug(f"Found file {filename}")
                name = filename.relative_to(d).with_suffix("")
                df_list[name] = pd.read_csv(filename, header=0, index_col=0)
            if not df_list:
                continue

            for name, df in df_list.items():
                p = name.parts
                # name is assumed to be cla/ssi/fier/features
                table[("/".join(p[:-1]), p[-1])] = (
                    df["uar"].mean(),
                    df["uar"].std(),
                    df["uar"].max(),
                )

            # multi-index, each value in table is a tuple of 3 floats
            df_uar = pd.DataFrame(
                table.values(), index=table.keys(), columns=["mean", "std", "max"]
            )
            # d.name is assumed to be the corpus name
            uar_list[d.name] = df_uar

        if not uar_list:
            raise FileNotFoundError("No valid files found matching regex.")

        # Concatenate DataFrames for all datasets
        df = pd.concat(uar_list, names=["Corpus", "Classifier", "Features"])
        df.to_csv("/tmp/emotion_results.csv")

    # Move corpus to columns, outermost level
    df = df.unstack(0).swaplevel(axis=1)
    df.columns.names = ["Corpus", "Metric"]

    if len(metrics) == 0:
        metrics = ("mean",)

    df = df.loc[:, (slice(None), metrics)]
    df = df.sort_index(axis=1, level=0)

    full = df
    df = df.loc[:, (slice(None), "mean")].droplevel("Metric", axis=1)

    rankfeat = sorted(df.index.get_level_values("Features").unique())
    rankclf = sorted(df.index.get_level_values("Classifier").unique())

    # Statistical tests
    if test is not None:
        flat = df.stack().reset_index().rename(columns={0: "UAR"})
        if test == "friedman":
            _table = flat.pivot_table(
                index="Corpus", columns="Classifier", values="UAR"
            )
            print("Friedman test for classifiers by corpus:")
            run_friedman(_table)
            avgrank = np.argsort(rankdata(-_table.to_numpy(), axis=1).mean(0))
            rankclf = _table.columns[avgrank]

            _table = flat.pivot_table(index="Corpus", columns="Features", values="UAR")
            print("Friedman test for features by corpus:")
            run_friedman(_table)
            avgrank = np.argsort(rankdata(-_table.to_numpy(), axis=1).mean(0))
            rankfeat = _table.columns[avgrank]
        else:
            run_anova(flat)

        if plot:
            plt.show()

    # Get the best (classifier, features) combination and corresponding UAR
    best = pd.concat([df.idxmax(0), df.max(0)], 1, keys=["Combination", "UAR"])
    best["Classifier"] = best["Combination"].map(lambda t: t[0])
    best["Features"] = best["Combination"].map(lambda t: t[1])
    best = best[["Classifier", "Features", "UAR"]]

    # Classifier-by-features table
    clf_feat = df.mean(1).unstack(1).loc[rankclf, rankfeat]

    # {mean, max} per {classifier, features}
    mean_clf = df.mean(level=0).T[rankclf]
    mean_feat = df.mean(level=1).T[rankfeat]
    max_clf = df.max(level=0).T[rankclf]
    max_feat = df.max(level=1).T[rankfeat]

    if substitute:
        substitute_labels(df)

        substitute_labels(best)
        best["Classifier"] = best["Classifier"].map(subs)
        best["Features"] = best["Features"].map(subs)

        # Order the columns how we want
        clf_feat = clf_feat.loc[
            ordered_intersect(SUBSTITUTIONS, clf_feat.index),
            ordered_intersect(SUBSTITUTIONS, clf_feat.columns),
        ]
        substitute_labels(clf_feat)
        mean_feat = mean_feat[ordered_intersect(SUBSTITUTIONS, mean_feat.columns)]
        substitute_labels(mean_feat)
        max_feat = max_feat[ordered_intersect(SUBSTITUTIONS, max_feat.columns)]
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

    if images or plot:
        mean_clf_fig = plot_matrix(mean_clf, cmap=cmap)
        max_clf_fig = plot_matrix(max_clf, cmap=cmap)
        mean_feat_fig = plot_matrix(mean_feat, cmap=cmap)
        max_feat_fig = plot_matrix(max_feat, cmap=cmap)
        clf_feat_fig = plot_matrix(clf_feat, cmap=cmap)

        if images:
            images.mkdir(exist_ok=True, parents=True)
            with PdfPages(images / "mean_clf.pdf") as pdf:
                pdf.savefig(mean_clf_fig)

            with PdfPages(images / "max_clf.pdf") as pdf:
                pdf.savefig(max_clf_fig)

            with PdfPages(images / "mean_feat.pdf") as pdf:
                pdf.savefig(mean_feat_fig)

            with PdfPages(images / "max_feat.pdf") as pdf:
                pdf.savefig(max_feat_fig)

            with PdfPages(images / "clf_feat.pdf") as pdf:
                pdf.savefig(clf_feat_fig)

        if plot:
            plt.show()

    if latex:
        latex.mkdir(exist_ok=True, parents=True)
        to_latex(
            full.stack(),
            latex / "combined.tex",
            label="tab:CombinedResults",
            caption="Combined results table",
        )
        to_latex(
            mean_clf,
            latex / "mean_clf.tex",
            label="tab:MeanClassifier",
            caption="Mean average UAR for each classifier.",
        )
        to_latex(
            mean_feat,
            latex / "mean_feat.tex",
            label="tab:MeanFeature",
            caption="Mean average UAR for each feature set.",
        )
        to_latex(
            max_clf,
            latex / "max_clf.tex",
            label="tab:MaxClassifier",
            caption="Maximum average UAR for each classifier.",
        )
        to_latex(
            max_feat,
            latex / "max_feat.tex",
            label="tab:MaxFeature",
            caption="Maximum average UAR for each feature set.",
        )
        to_latex(
            clf_feat,
            latex / "clf_feat.tex",
            label="tab:FeatClassifier",
            caption="Performance of (classifier, feature) pairs averaged"
            "across datasets",
        )

    if excel:
        for tab, tabname in zip(
            [full.stack(), clf_feat, mean_clf, mean_feat, max_clf, max_feat],
            ["combined", "clf_feat", "mean_clf", "mean_feat", "max_clf", "max_feat"],
        ):
            tab.to_excel(excel / (tabname + ".xlsx"), merge_cells=False)


if __name__ == "__main__":
    main()
