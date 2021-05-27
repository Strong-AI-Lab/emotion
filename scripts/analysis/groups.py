from pathlib import Path
from typing import Iterable, Tuple

import click
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from emorec.dataset import CombinedDataset, LabelledDataset
from emorec.stats import corr_ratio, dunn
from emorec.utils import PathlibPath


def get_combined_data(files: Iterable[Path]):
    datasets = []
    for file in files:
        print(f"Loading {file}")
        dataset = LabelledDataset(file)
        dataset_dir = Path("datasets", dataset.corpus)
        dataset.update_speakers(dataset_dir / "speaker.csv")
        dataset.update_labels(dataset_dir / "label.csv")
        sp_group_path = dataset_dir / "group.csv"
        if sp_group_path.exists():
            dataset.update_annotation("speaker_groups", sp_group_path)
        gender_path = dataset_dir / "gender.csv"
        if gender_path.exists():
            dataset.update_annotation("gender", gender_path)
        datasets.append(dataset)
    return datasets


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False), nargs=-1)
@click.option(
    "--transform",
    type=click.Choice(
        [
            "identity",
            "pca",
            "kpca",
            "lda",
            "fa",
            "ica",
            "umap_u",
            "umap_s",
            "tsne",
            "rp",
            "isomap",
        ]
    ),
    help="Transformation to apply before clustering and plotting.",
)
@click.option(
    "--std/--nostd", default=True, help="Standardise features.", show_default=True
)
@click.option(
    "--part", "--groups", default="speakers", help="Partition to use for comparison."
)
@click.option("--plot/--noplot", default=False, help="2D plot of feature space.")
@click.option("--metric", default="l2", help="Distance metric to use.")
@click.option("--kernel", default="rbf", help="Kernel to use for kernel PCA.")
@click.option("--csv", type=Path, help="Output CSV with metrics.")
def main(
    input: Tuple[Path],
    transform: str,
    std: bool,
    part: str,
    plot: bool,
    metric: str,
    kernel: str,
    csv: Path,
):
    print("Combining data.")
    data = CombinedDataset(*get_combined_data(input))

    # Note this is not specifically speaker groups
    group_indices = data.get_group_indices(part)
    group_names = list(data.partitions[part].keys())
    n_groups = group_indices.max() + 1

    print(f"{n_groups} groups:")
    print(", ".join(group_names))

    if std:
        print("Standardising features independently (globally)")
        data.normalise(StandardScaler(), scheme="all")
    x = data.x

    print(f"Transforming with {transform}")
    if transform == "pca":
        x = PCA().fit_transform(x)
    elif transform == "kpca":
        x = KernelPCA(kernel=kernel, n_jobs=-1).fit_transform(x)
    elif transform == "lda":
        x = LinearDiscriminantAnalysis().fit_transform(x, group_indices)
    elif transform == "fa":
        x = FactorAnalysis(max_iter=500).fit_transform(x)
    elif transform == "ica":
        x = FastICA(max_iter=100).fit_transform(x)
    elif transform in ["umap_u", "umap_s"]:
        # Import here due to loading time
        from umap import UMAP

        umap = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric=metric)
        if transform == "umap_u":
            x = umap.fit_transform(x)
        elif transform == "umap_s":
            x = umap.fit_transform(x, group_indices)
    elif transform == "tsne":
        x = TSNE(perplexity=50, metric=metric, n_jobs=-1).fit_transform(x)
    elif transform == "rp":
        x = GaussianRandomProjection(2).fit_transform(x)
    elif transform == "isomap":
        x = Isomap(n_neighbors=5, metric=metric, n_jobs=-1).fit_transform(x)

    print("Calculating intra-cluster distances")
    intra_dists = []
    for i in range(n_groups):
        subset = x[group_indices == i]
        intra_dists.append(pairwise_distances(subset, metric=metric, n_jobs=-1).mean())

    df = pd.DataFrame(
        {
            "corpora": " ".join(data.corpora),
            "partition": part,
            "transform": transform,
            "metric": metric,
            "kernel": kernel,
            "standardisation": std,
            "dunn_index": dunn(x, group_indices, metric=metric),
            "mean_correlation_ratio": corr_ratio(x, group_indices).mean(),
            "davies_bouldin": davies_bouldin_score(x, group_indices),
            "silhouette": silhouette_score(x, group_indices, metric=metric),
            "calinski_harabasz_score": calinski_harabasz_score(x, group_indices),
            "min_intra_dist": np.min(intra_dists),
            "max_intra_dist": np.max(intra_dists),
            "mean_intra_dist": np.mean(intra_dists),
            "std_intra_dist": np.std(intra_dists),
        },
        index=[0],
    )

    if csv:
        csv.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(csv, index=False)
        print(f"Wrote CSV to {csv}")
    else:
        print(df.loc[0].to_string())

    if plot:
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_opts = dict(marker=".", s=1)
        cmap = cm.get_cmap("rainbow")
        ax.set_title(f"Instance distribution by {part}")
        ax.set_xticks([])
        ax.set_yticks([])
        colours = cmap(np.linspace(0, 1, n_groups))
        for grp in range(n_groups):
            grp_data = x[group_indices == grp]
            ax.scatter(
                grp_data[:, 0],
                grp_data[:, 1],
                color=colours[grp],
                label=group_names[grp],
                **plot_opts,
            )
        ax.legend()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
