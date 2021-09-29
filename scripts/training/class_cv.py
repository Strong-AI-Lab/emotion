import json
import warnings
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from click_option_group import optgroup
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ertk.classification import within_corpus_cross_validation
from ertk.dataset import load_multiple
from ertk.utils import (
    PathlibPath,
    SequenceTransformWrapper,
    get_arg_mapping,
    get_cv_splitter,
)


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False), nargs=-1)
@click.option("--features", required=True, help="Features to load.")
@click.option("--clf", "clf_type", required=True, help="Classifier to use.")
@optgroup.group("Results options")
@optgroup.option("--results", type=Path, help="Results directory.")
@optgroup.option("--logdir", type=Path, help="TF/PyTorch logs directory.")
@optgroup.group("Data options")
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
@optgroup.group("Cross-validation options")
@optgroup.option("--partition", help="Partition for LOGO CV.")
@optgroup.option(
    "--kfold",
    type=int,
    default=5,
    help="k when using (group) k-fold cross-validation, or leave-one-out.",
)
@optgroup.option(
    "--inner_kfold",
    type=int,
    default=2,
    help="k for inner k-fold CV (where relevant). If None then LOGO is used. "
    "If 1 then a random split is used.",
)
@optgroup.option("--test_size", type=float, help="Test size when kfold=1.")
@optgroup.option(
    "--inner_group/--noinner_group",
    default=True,
    help="Whether to use group-based inner CV (e.g. GroupKFold, LeaveOneGroupOut).",
)
@optgroup.group("Training options")
@optgroup.option("--learning_rate", type=float, default=1e-4, show_default=True)
@optgroup.option("--batch_size", type=int, default=64, show_default=True)
@optgroup.option("--epochs", type=int, default=50, show_default=True)
@optgroup.option(
    "--balanced/--imbalanced", default=True, help="Balances sample weights."
)
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
    "--n_jobs", type=int, default=-1, help="Number of parallel executions."
)
@optgroup.group("Misc. options")
@optgroup.option("--verbose", count=True, help="Verbose training.")
@optgroup.group("Model-specific options")
@optgroup.option(
    "--clf_args",
    "clf_args_file",
    type=PathlibPath(exists=True, dir_okay=False),
    multiple=True,
    help="File containing keyword arguments to give to model initialisation.",
)
@optgroup.option(
    "--param_grid",
    "param_grid_file",
    type=PathlibPath(exists=True),
    multiple=True,
    help="File with parameter grid data.",
)
@optgroup.group(
    "Deprecated options", help="These options are deprecated, don't use them."
)
@optgroup.option(
    "--inner_cv/--noinner_cv",
    "use_inner_cv",
    default=True,
    help="Whether to use inner CV. This is deprecated and only exists for backwards "
    "compatibility.",
)
def main(
    clf_type: str,
    input: Tuple[Path],
    features: str,
    partition: str,
    kfold: int,
    inner_kfold: int,
    test_size: float,
    inner_group: bool,
    results: Path,
    logdir: Path,
    reps: int,
    normalise: str,
    transform: str,
    balanced: bool,
    map_groups: Tuple[str],
    sel_groups: Tuple[str],
    clip_seq: int,
    pad_seq: int,
    verbose: int,
    clf_args_file: Tuple[Path],
    param_grid_file: Tuple[Path],
    learning_rate: float,
    batch_size: int,
    epochs: int,
    use_inner_cv: bool,
    n_jobs: int,
):
    """Runs cross-validation on the given INPUT datasets. Metrics are
    optionally written to a results file.
    """

    if verbose > 0:
        import logging

        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)

    dataset = load_multiple(input, features)

    for m in map_groups:
        mapping = get_arg_mapping(m)
        for part in mapping:
            dataset.map_groups(part, mapping[part])
    for m in sel_groups:
        mapping = get_arg_mapping(m)
        for part in mapping:
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

    cv = get_cv_splitter(bool(partition), kfold, test_size=test_size)
    if not use_inner_cv:
        warnings.warn(
            "--noinner_cv is deprecated and only exists for backwards compatibility."
        )
        inner_cv = cv
    else:
        inner_cv = get_cv_splitter(inner_group, inner_kfold)

    sample_weight = None
    if balanced:
        class_weight = dataset.n_instances / (dataset.n_classes * dataset.class_counts)
        sample_weight = class_weight[dataset.y]
    fit_params = {"sample_weight": sample_weight}

    model_args = {}
    for file in clf_args_file:
        model_args.update(get_arg_mapping(file))
    param_grid = {}
    for file in param_grid_file:
        param_grid.update(get_arg_mapping(file))

    clf_lib, clf_type = clf_type.split("/")
    if clf_lib == "sk":
        from sklearn.model_selection import GridSearchCV

        from ertk.sklearn.models import get_sk_model

        clf = get_sk_model(clf_type, **model_args)
        clf = GridSearchCV(
            Pipeline([("transform", transformer), ("clf", clf)]),
            {"clf__" + k: v for k, v in param_grid.items()},
            scoring="balanced_accuracy",
            cv=inner_cv,
            verbose=verbose,
            n_jobs=1 if use_inner_cv else n_jobs,
        )
        params = param_grid
        if not use_inner_cv:
            # Get best params then pass best to main cross-validation routine
            # XXX: This exists because of mistake in our original implementation.
            clf.fit(
                dataset.x,
                dataset.labels,
                groups=dataset.get_group_indices(partition) if partition else None,
                clf__sample_weight=sample_weight,
            )
            params = clf.best_params_
            clf = clf.best_estimator_
    elif clf_lib == "tf":
        from tensorflow.keras.optimizers import Adam

        from ertk.tensorflow import get_tf_model
        from ertk.tensorflow.classification import tf_classification_metrics

        # No parallel CV to avoid running out of GPU memory
        n_jobs = 1
        fit_params.update(
            {
                "log_dir": logdir,
                "epochs": epochs,
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
            **model_args,
            **fit_params,
        }
        clf = model_fn
    else:
        raise ValueError(f"Invalid classifier type {clf_type}")

    print("== Dataset ==")
    print(dataset)
    print("== Cross-validation settings ==")
    print(f"Using {dataset.n_classes}-class classification.")
    print(f"Using classifier {clf}.")
    print(f"Using {kfold} k-fold CV.")
    if partition:
        print(f"Using {partition} as split partition.")
    if use_inner_cv:
        print(f"Using {inner_cv} as inner CV splitter.")
    if balanced:
        print("Using balanced sample weights.")
    if normalise == "online":
        print("Using 'online' normalisation.")
    else:
        print(f"Normalising globally using {normalise} partition.")

    dfs = []
    for rep in range(1, reps + 1):
        print(f"Rep {rep}/{reps}")
        df = within_corpus_cross_validation(
            clf,
            dataset,
            clf_lib=clf_lib,
            partition=partition,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
            fit_params=fit_params,
        )
        df["params"] = [json.dumps(params, default=str)] * len(df)
        dfs.append(df)
    df = pd.concat(dfs, keys=list(range(1, reps + 1)), names=["rep"])
    df["clf"] = f"{clf_lib}/{clf_type}"
    df["corpus"] = dataset.corpus
    df["features"] = Path(features).stem
    if results:
        results.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results)
        print(f"Wrote CSV to {results}")
    else:
        print(df)
    print(df.mean(numeric_only=True))


if __name__ == "__main__":
    main()
