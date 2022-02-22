import json
import warnings
from functools import partial
from pathlib import Path
from typing import Tuple

import click
import numpy as np
from click_option_group import optgroup
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ertk.classification import get_balanced_sample_weights, train_val_test
from ertk.config import get_arg_mapping
from ertk.dataset import load_multiple
from ertk.train import ValidationSplit
from ertk.transform import SequenceTransformWrapper


@click.command()
@click.argument(
    "input", type=click.Path(exists=True, dir_okay=False, path_type=Path), nargs=-1
)
@click.option("--features", required=True, help="Features to load.")
@click.option("--clf", "clf_type", required=True, help="Classifier to use.")
@click.option("--train", required=True, help="Train data.")
@click.option("--valid", required=True, help="Validation data.")
@click.option("--test", required=True, help="Test data.")
@optgroup.group("Results options")
@optgroup.option("--results", type=Path, help="Results directory.")
@optgroup.option("--logdir", type=Path, help="TF/PyTorch logs directory.")
@optgroup.group("Data options")
@optgroup.option("--label", default="label", help="Label annotation to use.")
@optgroup.option(
    "--subset", multiple=True, default=["default"], help="Subset selection."
)
@optgroup.option("--map_groups", multiple=True, help="Group mapping.")
@optgroup.option(
    "--sel_groups",
    multiple=True,
    help="Group selection. This is a map from partition to group.",
)
@optgroup.option(
    "--clip_seq", type=int, help="Clip sequences to this length (before pad)."
)
@optgroup.option(
    "--pad_seq", type=int, help="Pad sequences to multiple of this length (after clip)."
)
@optgroup.group("Experiment options")
@optgroup.option(
    "--reps",
    type=int,
    default=1,
    show_default=True,
    help="The number of repetitions to do per test.",
)
@optgroup.option(
    "--normalise",
    default="online",
    show_default=True,
    help="Normalisation method. 'online' means use training data for normalisation.",
)
@optgroup.option(
    "--transform",
    type=click.Choice(["std", "minmax"]),
    default="std",
    show_default=True,
    help="Transformation class.",
)
@optgroup.option(
    "--balanced/--imbalanced", default=True, help="Balances sample weights."
)
@optgroup.group("Misc. options")
@optgroup.option(
    "--verbose",
    type=int,
    default=0,
    help="Verbosity. -1=nothing, 0=dataset+results, 1=INFO, 2=DEBUG",
)
@optgroup.group("Model-specific options")
@optgroup.option(
    "--clf_args",
    "clf_args_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="File containing keyword arguments to give to model initialisation.",
)
@optgroup.option(
    "--param_grid",
    "param_grid_file",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    help="File with parameter grid data.",
)
@optgroup.option("--learning_rate", type=float, default=1e-4, show_default=True)
@optgroup.option("--batch_size", type=int, default=64, show_default=True)
@optgroup.option("--epochs", type=int, default=50, show_default=True)
def main(
    clf_type: str,
    input: Tuple[Path],
    features: str,
    train: str,
    valid: str,
    test: str,
    label: str,
    param_grid_file: Tuple[Path],
    results: Path,
    logdir: Path,
    reps: int,
    normalise: str,
    transform: str,
    balanced: bool,
    subset: Tuple[str],
    map_groups: Tuple[str],
    sel_groups: Tuple[str],
    clip_seq: int,
    pad_seq: int,
    verbose: int,
    clf_args_file: Tuple[Path],
    learning_rate: float,
    batch_size: int,
    epochs: int,
):
    """Runs training on the given INPUT dataset, using training set.
    Validation is done on validation set and results reported on test
    set. Metrics are optionally written to a results file.
    """

    np.set_printoptions(precision=4, threshold=10)
    if verbose > 0:
        import logging

        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)
        np.set_printoptions()

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

    sample_weight = None
    if balanced:
        sample_weight = get_balanced_sample_weights(dataset.get_annotations(label))
    fit_params = {"sample_weight": sample_weight}

    model_args = {}
    for file in clf_args_file:
        model_args.update(get_arg_mapping(file))
    param_grid = {}
    for file in param_grid_file:
        param_grid.update(get_arg_mapping(file))

    train_indices = dataset.get_idx_for_split(train)
    valid_indices = dataset.get_idx_for_split(valid)
    test_indices = dataset.get_idx_for_split(test)

    clf_lib, clf_type = clf_type.split("/", maxsplit=1)
    if clf_lib == "sk":
        from sklearn.base import clone
        from sklearn.model_selection import GridSearchCV

        from ertk.sklearn.models import get_sk_model

        class GridSearchWrapper(GridSearchCV):
            """Simple hack to avoid retraining on train + val data."""

            def fit(self, X, y=None, *, groups=None, **fit_params):
                self.refit = False
                super().fit(X, y=y, groups=groups, **fit_params)
                self.best_estimator_ = clone(
                    clone(self.estimator).set_params(**self.best_params_)
                )
                train, _ = next(iter(self.cv.split(X, y, groups)))
                self.best_estimator_.fit(X[train], y[train], **fit_params)
                return self.best_estimator_

        clf = get_sk_model(clf_type, **model_args)
        clf = GridSearchWrapper(
            Pipeline([("transform", transformer), ("clf", clf)]),
            {f"clf__{k}": v for k, v in param_grid.items()},
            scoring="balanced_accuracy",
            cv=ValidationSplit(
                np.arange(len(train_indices), len(train_indices) + len(valid_indices))
            ),
            verbose=max(verbose, 0),
            n_jobs=-1,
            refit=False,
        )
        params = param_grid
        # A hack so that the ValidationSplit works properly.
        train_indices = np.concatenate([train_indices, valid_indices])
    elif clf_lib == "tf":
        from tensorflow.keras.callbacks import TensorBoard
        from tensorflow.keras.optimizers import Adam

        from ertk.tensorflow import get_tf_model
        from ertk.tensorflow.classification import tf_classification_metrics

        callbacks = []
        if logdir is not None:
            callbacks.append(
                TensorBoard(
                    log_dir=logdir,
                    profile_batch=0,
                    write_graph=False,
                    write_images=False,
                )
            )
        fit_params.update(
            {
                "epochs": epochs,
                "callbacks": callbacks,
                "batch_size": batch_size,
                "data_fn": None,
            }
        )
        model_args.update(
            {"n_features": dataset.n_features, "n_classes": dataset.n_classes}
        )

        def model_fn():
            model = get_tf_model(clf_type, **model_args)
            model.compile(
                Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=tf_classification_metrics(),
            )
            return Pipeline([("transform", transformer), ("clf", model)])

        params = {
            "optimiser": "adam",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            **model_args,
            **fit_params,
        }
        clf = model_fn
    elif clf_lib == "pt":
        import torch.optim

        from ertk.pytorch import get_pt_model

        optim_fn = partial(torch.optim.Adam, lr=learning_rate)
        fit_params.update(
            {
                "log_dir": logdir,
                "max_epochs": epochs,
                "batch_size": batch_size,
                "optim_fn": optim_fn,
            }
        )
        model_args.update(
            {"n_features": dataset.n_features, "n_classes": dataset.n_classes}
        )

        def model_fn():
            model = get_pt_model(clf_type, **model_args)
            return model if transformer is None else (transformer, model)

        params = {
            "optimiser": "adam",
            "learning_rate": learning_rate,
            **model_args,
            **fit_params,
        }
        clf = model_fn
    else:
        raise ValueError(f"Invalid classifier type {clf_type}")

    if verbose > -1:
        print("== Dataset ==")
        print(dataset)
        print("== Train/valid/test settings ==")
        print(f"Using {dataset.n_classes}-class classification.")
        print(f"Using classifier {clf}.")
        print(f"Using fit_params={fit_params}")
        print(f"Training set: {train}")
        print(f"Validation set: {valid}")
        print(f"Test set: {test}")
        if balanced:
            print("Using balanced sample weights.")
        if normalise == "online":
            print("Using 'online' normalisation.")
        else:
            print(f"Normalising globally using {normalise} partition.")

    df = train_val_test(
        clf,
        dataset=dataset,
        train_idx=train_indices,
        valid_idx=valid_indices,
        test_idx=test_indices,
        label=label,
        clf_lib=clf_lib,
        sample_weight=sample_weight,
        verbose=verbose,
        fit_params=fit_params,
    )
    df["clf"] = f"{clf_lib}/{clf_type}"
    df["features"] = features
    df["corpus"] = dataset.corpus
    df["params"] = json.dumps(params, default=str)
    if results:
        results.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results)
        print(f"Wrote CSV to {results}")
    else:
        print(df)
    if verbose > -1:
        print(df.mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
