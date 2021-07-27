import json
from functools import partial
from pathlib import Path
from typing import Callable, Tuple

import click
from click_option_group import optgroup
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emorec.classification import within_corpus_cross_validation
from emorec.dataset import load_multiple
from emorec.sklearn.models import (
    PrecomputedSVC,
    default_rf_param_grid,
    default_svm_param_grid,
)
from emorec.tensorflow import compile_wrap, get_tf_model_fn
from emorec.tensorflow.models.mlp import model as mlp_model
from emorec.utils import PathlibPath, get_cv_splitter


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False), nargs=-1)
@click.option("--features", required=True, help="Features to load.")
@click.option("--clf", "clf_type", required=True, help="Classifier to use.")
@optgroup.group("Results options")
@optgroup.option("--results", type=Path, help="Results directory.")
@optgroup.group("Cross-validation options")
@optgroup.option("--partition", help="Partition for LOGO CV.")
@optgroup.option(
    "--kfold",
    type=int,
    default=-1,
    help="Optional k when using (group) k-fold cross-validation.",
)
@optgroup.option(
    "--inner_part", help="Partition to use for inner CV (e.g. with grid-search)."
)
@optgroup.option(
    "--inner_kfold",
    type=int,
    default=2,
    help="k for inner k-fold CV. If None then LOGO is used. If 1 then a random split "
    "is used.",
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
    "--sample_weight/--nosample_weight", default=True, help="Balances sample weights."
)
@optgroup.group("Misc. options")
@optgroup.option("--verbose/--noverbose", default=False, help="Verbose training.")
@optgroup.group("Model-specific options")
@optgroup.option("--learning_rate", type=float, default=1e-4, show_default=True)
@optgroup.option("--batch_size", type=int, default=64, show_default=True)
@optgroup.option("--epochs", type=int, default=50, show_default=True)
def main(
    clf_type: str,
    input: Tuple[Path],
    features: str,
    partition: str,
    kfold: int,
    inner_part: str,
    inner_kfold: int,
    results: Path,
    reps: int,
    normalise: str,
    sample_weight: bool,
    verbose: bool,
    learning_rate: float,
    batch_size: int,
    epochs: int,
):
    """Runs a training routine using the given classifier CLF_TYPE and
    INPUT datasets. Metrics are optionally written to a results file.
    """

    if len(input) == 0:
        raise ValueError("No input dataset(s) specified.")

    dataset = load_multiple(input, features)
    if normalise == "online":
        transform = StandardScaler()
    else:
        dataset.normalise(normaliser=StandardScaler(), partition=normalise)
        transform = None

    splitter = get_cv_splitter(bool(partition), kfold)

    sample_weights = None
    if sample_weight:
        class_weight = dataset.n_instances / (dataset.n_classes * dataset.class_counts)
        sample_weights = class_weight[dataset.y]

    n_jobs = -1
    if clf_type.startswith(("svm", "rf")):
        if clf_type.startswith("svm"):
            param_grid = default_svm_param_grid(clf_type.split("/")[-1])
            cls = PrecomputedSVC
        else:
            param_grid = default_rf_param_grid()
            cls = RandomForestClassifier
        param_grid = {"clf__" + k: v for k, v in param_grid.items()}
        inner_cv = get_cv_splitter(inner_part is not None, inner_kfold)
        clf = GridSearchCV(
            Pipeline([("transform", transform), ("clf", cls())]),
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            verbose=verbose,
            n_jobs=1,
        )
        params = param_grid
    else:
        import tensorflow as tf

        tf.get_logger().setLevel(40)  # ERROR level
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        if clf_type.startswith("mlp"):
            _layers = clf_type.split("/")[-1]
            if _layers == "1layer":
                layers = 1
                # Force CPU only for 1 layer, empirically faster
                import os

                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            elif _layers == "2layer":
                layers = 2
                n_jobs = 1
            elif _layers == "3layer":
                layers = 3
                n_jobs = 1
            model_fn: Callable = partial(mlp_model, units=(512,) * layers)
        else:
            model_fn = get_tf_model_fn(clf_type)
        params = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        clf = KerasClassifier(
            compile_wrap(model_fn, opt_kwargs=dict(learning_rate=learning_rate)),
            n_features=dataset.n_features,
            n_classes=dataset.n_classes,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
        )
        clf = Pipeline([("transform", transform), ("clf", clf)])

    print("== Dataset ==")
    print(dataset)
    print()
    print("== Cross-validation settings ==")
    print(f"Using {dataset.n_classes}-class classification.")
    print(f"Using {splitter} as CV splitter.")
    if partition:
        print(f"Using {partition} as split partition.")
    print(f"Using {inner_cv} as inner CV splitter.")
    if inner_part:
        print(f"Using {inner_part} as inner CV partition.")
    if sample_weight:
        print("Using balanced sample weights.")
    if normalise == "online":
        print("Using 'online' normalisation.")
    else:
        print(f"Normalising globally using {normalise} partition.")

    df = within_corpus_cross_validation(
        clf,
        dataset,
        partition=partition,
        reps=reps,
        sample_weights=sample_weights,
        splitter=splitter,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    df["params"] = [json.dumps(params, default=list)] * len(df)
    if results:
        results.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results)
        print(f"Wrote CSV to {results}")
    else:
        print(df)
        print(df.mean(numeric_only=True))


if __name__ == "__main__":
    main()
