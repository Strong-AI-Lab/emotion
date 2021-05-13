from pathlib import Path
from typing import Iterable, Tuple

import click
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from emorec.dataset import CombinedDataset, LabelledDataset
from emorec.stats import corr_ratio, dunn, silhouette
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
        ["pca", "kpca", "lda", "fa", "ica", "umap_u", "umap_s", "tsne", "rp", "isomap"]
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
def main(
    input: Tuple[Path],
    transform: str,
    std: bool,
    part: str,
    plot: bool,
    metric: str,
    kernel: str,
):
    print("Combining data.")
    data = CombinedDataset(*get_combined_data(input))

    # Note this is not specifically speaker groups
    group_indices = data.get_group_indices(part)
    group_names = list(data.partitions[part].keys())
    n_groups = len(data.partitions[part])

    print(f"{n_groups} groups:")
    print(", ".join(group_names))

    if std:
        print("Standardising features independently (globally)")
        data.normalise(StandardScaler(), scheme="all")
    x = data.x

    if transform == "pca":
        x = PCA().fit_transform(x)
    elif transform == "kpca":
        x = KernelPCA(kernel=kernel).fit_transform(x)
    elif transform == "lda":
        x = LinearDiscriminantAnalysis().fit_transform(x, group_indices)
    elif transform == "fa":
        x = FactorAnalysis().fit_transform(x)
    elif transform == "ica":
        x = FastICA().fit_transform(x)
    elif transform.startswith("umap"):
        # Import here due to loading time
        from umap import UMAP

        umap = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric=metric)
        if transform == "umap_u":
            x = umap.fit_transform(x)
        elif transform == "umap_s":
            x = umap.fit_transform(x, group_indices)
    elif transform == "tsne":
        x = TSNE(perplexity=50, metric=metric).fit_transform(x)
    elif transform == "rp":
        x = GaussianRandomProjection(2).fit_transform(x)
    elif transform == "isomap":
        x = Isomap(n_neighboursmetric=metric).fit_transform(x)

    print(f"Dunn index {dunn(x, group_indices, metric=metric)}")
    print(f"Mean correlation ratio: {corr_ratio(x, group_indices).mean()}")
    print(f"Mean silhouette: {silhouette(x, group_indices, metric=metric)}")

    if plot:
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_opts = dict(marker=".", linewidths=0)
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
