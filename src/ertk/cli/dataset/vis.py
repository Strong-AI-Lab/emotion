from pathlib import Path
from typing import Tuple, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from umap import UMAP

from ertk.dataset import DataLoadConfig, load_datasets_config, read_features


@click.group()
def main():
    """Tools for visualising data."""


@main.command("xy")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--transform",
    type=click.Choice(["pca", "umap", "tsne", "std", "minmax"]),
    default=("pca",),
    multiple=True,
)
@click.option("--colour", "colour_part", type=str, help="Colour by this partition.")
@click.option(
    "--sample",
    "sample_str",
    type=str,
    default="1.0",
    help="Proportion or number of instances to sample for visualisation",
)
@click.argument("restargs", type=str, nargs=-1)
def xy(
    input: Path,
    transform: Tuple[str],
    colour_part: str,
    sample_str: str,
    restargs: Tuple[str],
):
    """Plots INPUT in 2D using one of a number of transforms. INPUT can
    either be a features file or data load config.
    """

    print(f"Loading data from {input}")
    if input.suffix == ".yaml":
        conf = DataLoadConfig.from_file(input, override=list(restargs))
        data = load_datasets_config(conf)

        x = data.x
        hue = data.get_annotations(colour_part)
    else:
        x = read_features(input).features
        hue = None

    try:
        sample = int(float(sample_str) * len(x))
    except ValueError:
        sample = int(sample_str)
    print(f"Using {sample} instances for visualisation")
    if sample < len(x):
        rng = np.random.default_rng()
        perm = rng.permutation(len(x))[:sample]
        x = x[perm]
        if hue is not None:
            hue = hue[perm]

    trn = []
    for trans in transform:
        if trans == "pca":
            trn.append(PCA(n_components=2))
        elif trans == "tsne":
            trn.append(TSNE(n_components=2))
        elif trans == "minmax":
            trn.append(MinMaxScaler())
        elif trans == "std":
            trn.append(StandardScaler())
        elif trans == "norm":
            trn.append(Normalizer())
        elif trans == "umap":
            trn.append(UMAP(n_components=2))
    pipeline = make_pipeline(*trn)

    print(f"Transforming with {pipeline}")
    x_new = pipeline.fit_transform(x)
    plt.figure()
    sns.scatterplot(x=x_new[:, 0], y=x_new[:, 1], hue=hue)
    plt.show()


@main.command("feats")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("instance", type=str, default="2")
def feats(input: Path, instance: str):
    """Displays plot of INSTANCE in INPUT. INSTANCE can either be a
    numeric index, a range of indices using numpy slice notation or a
    named instance.
    """

    data = read_features(input)
    if instance.isdigit():
        idx: Union[int, slice] = int(instance)
    else:
        _i = instance.find(":")
        if _i != -1:
            start = int(instance[:_i])
            end = int(instance[_i + 1 :])
            idx = slice(start, end)
        else:
            idx = data.names.index(instance)
    arr = data.features[idx]
    names = data.names[idx]
    print(names)

    plt.figure()
    plt.imshow(arr, aspect="equal", origin="upper", interpolation="nearest")
    plt.xlabel("Features")
    plt.ylabel("Instance" if len(names) > 1 else "Time")
    plt.show()


if __name__ == "__main__":
    main()
