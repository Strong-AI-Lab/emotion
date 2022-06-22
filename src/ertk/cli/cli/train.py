import pickle
import warnings
from pathlib import Path
from typing import Tuple

import click
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ertk.classification import get_balanced_sample_weights
from ertk.config import get_arg_mapping
from ertk.dataset import load_multiple
from ertk.sklearn.models import get_sk_model
from ertk.transform import SequenceTransformWrapper


@click.command()
@click.argument(
    "input", type=click.Path(exists=True, dir_okay=False, path_type=Path), nargs=-1
)
@click.option("--features", required=True, help="Features to load.")
@click.option("--clf", "clf_type", required=True, help="Classifier to use.")
@click.option("--save", type=Path, required=True, help="Location to save the model.")
@click.option(
    "--cv",
    type=click.Choice(["speaker", "corpus"]),
    help="Cross-validation method.",
)
@click.option("--balanced/--imbalanced", default=True, help="Balances sample weights.")
@click.option(
    "--normalise",
    type=click.Choice(["speaker", "corpus", "online"]),
    default="speaker",
    help="Normalisation method.",
)
@click.option(
    "--transform",
    type=click.Choice(["std", "minmax"]),
    default="std",
    show_default=True,
    help="Transformation class.",
)
@click.option("--subset", multiple=True, default=["default"], help="Subset selection.")
@click.option("--map_groups", multiple=True, help="Group name mapping.")
@click.option(
    "--sel_groups",
    multiple=True,
    help="Group selection. This is a map from partition to group(s).",
)
@click.option(
    "--clip_seq", type=int, help="Clip sequences to this length (before pad)."
)
@click.option(
    "--pad_seq", type=int, help="Pad sequences to multiple of this length (after clip)."
)
@click.option("--target", default="label", help="Classifier/regressor target.")
@click.option(
    "--clf_args",
    "clf_args_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File containing keyword arguments to give to model initialisation.",
)
@click.option(
    "--param_grid",
    "param_grid_file",
    type=click.Path(exists=True, path_type=Path),
    help="File with parameter grid data.",
)
@click.option("--verbose", count=True, help="Verbose training.")
def main(
    input: Tuple[Path],
    features: str,
    clf_type: str,
    save: Path,
    cv: str,
    normalise: str,
    transform: str,
    balanced: bool,
    subset: Tuple[str],
    map_groups: Tuple[str],
    sel_groups: Tuple[str],
    clip_seq: int,
    pad_seq: int,
    target: str,
    clf_args_file: Path,
    param_grid_file: Path,
    verbose: int,
):
    """Trains a model on the given INPUT datasets. INPUT files must be
    corpus info files.

    Optionally pickles the model.
    """

    dataset = load_multiple(input, features)

    if len(subset) == 1:
        dataset = load_multiple(input, features, subsets=subset[0])
    else:
        subset_map = {}
        for m in subset:
            subset_map.update(get_arg_mapping(m))
        dataset = load_multiple(input, features, subsets=subset_map)

    for m in map_groups:
        mapping = get_arg_mapping(m)
        for part in mapping:
            if part not in dataset.partitions:
                warnings.warn(
                    f"Partition {part} cannot be mapped as it is not in the dataset."
                )
                continue
            dataset.map_groups(part, mapping[part])
    for m in sel_groups:
        mapping = get_arg_mapping(m)
        for part in mapping:
            if part not in dataset.partitions:
                warnings.warn(
                    f"Partition {part} cannot be selected as it is not in the dataset."
                )
                continue
            keep = mapping[part]
            if isinstance(keep, str):
                keep = [keep]
            dataset.remove_groups(part, keep=keep)

    if clip_seq:
        dataset.clip_arrays(clip_seq)
    if pad_seq:
        dataset.pad_arrays(pad_seq)

    transformer = {"std": StandardScaler, "minmax": MinMaxScaler}[transform]()
    if normalise == "none":
        transformer = None
    elif normalise == "online" and len(dataset.x[0].shape) > 1:
        transformer = SequenceTransformWrapper(transformer, "feature")
    elif normalise != "online":
        dataset.normalise(normaliser=transformer, partition=normalise)
        transformer = None

    print(dataset)

    sample_weight = None
    if balanced and target in dataset.partitions:
        sample_weight = get_balanced_sample_weights(dataset.get_annotations(target))
    fit_params = {"sample_weight": sample_weight}

    model_args = {}
    if clf_args_file:
        model_args = get_arg_mapping(clf_args_file)
    param_grid = {}
    if param_grid_file:
        param_grid = get_arg_mapping(param_grid_file)

    clf_lib, clf_type = clf_type.split("/")
    if clf_lib == "sk":
        model_args["probability"] = True
        clf = get_sk_model(clf_type, **model_args)
        clf = Pipeline([("transform"), ("clf", clf)])
        if cv:
            groups = dataset.get_group_indices(cv)
            clf = GridSearchCV(
                clf,
                param_grid,
                scoring="balanced_accuracy",
                cv=GroupKFold(2),
                n_jobs=-1,
                verbose=verbose,
            )
            fit_params["groups"] = groups

        y = dataset.get_annotations(target)
        clf.fit(dataset.x, y, **fit_params)
        save.parent.mkdir(parents=True, exist_ok=True)
        with open(save, "wb") as fid:
            pickle.dump(clf.best_estimator_, fid)
            print(f"Saved classifier to {save}")


if __name__ == "__main__":
    main()
