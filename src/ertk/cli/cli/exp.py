import json
import warnings
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ertk.classification import (
    dataset_cross_validation,
    get_balanced_sample_weights,
    train_val_test,
)
from ertk.cli._utils import dataset_args, eval_args, model_args, result_args, train_args
from ertk.config import get_arg_mapping
from ertk.dataset import load_multiple
from ertk.sklearn.utils import GridSearchVal
from ertk.train import ValidationSplit, get_cv_splitter
from ertk.transform import SequenceTransformWrapper


@click.command()
@dataset_args
@eval_args
@model_args
@result_args
@train_args
def main(
    clf_type: str,
    corpus_info: Tuple[Path],
    features: str,
    cv_part: str,
    label: str,
    kfold: int,
    inner_kfold: int,
    test_size: float,
    inner_part: str,
    results: Path,
    logdir: Path,
    reps: int,
    normalise: str,
    transform: str,
    balanced: bool,
    subset: Tuple[str],
    map_groups: Tuple[str],
    sel_groups: Tuple[str],
    remove_groups: Tuple[str],
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
    n_gpus: int,
    train: str,
    valid: str,
    test: str,
):
    np.set_printoptions(precision=4, threshold=10)
    if verbose > 0:
        import logging

        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)
        np.set_printoptions()

    if (train and kfold) or (not train and not kfold):
        raise ValueError("Exactly one of train and kfold must be given.")

    if len(subset) == 1 and ":" not in subset[0]:
        dataset = load_multiple(corpus_info, features, subsets=subset[0])
    else:
        subset_map = {}
        for m in subset:
            subset_map.update(get_arg_mapping(m))
        dataset = load_multiple(corpus_info, features, subsets=subset_map)

    grp_map = {}
    for m in map_groups:
        grp_map.update(get_arg_mapping(m))
    grp_sel = {}
    for m in sel_groups:
        grp_sel.update(get_arg_mapping(m))
    grp_del = {}
    for m in remove_groups:
        grp_del.update(get_arg_mapping(m))
    dataset.map_and_select(grp_map, grp_sel, grp_del)

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

    if not train:
        cv = get_cv_splitter(bool(cv_part), kfold, test_size=test_size)
        if not use_inner_cv:
            warnings.warn(
                "--noinner_cv is deprecated and only exists for backwards "
                "compatibility."
            )
            inner_cv = cv
        else:
            inner_cv = get_cv_splitter(bool(inner_part), inner_kfold)
    else:
        if not valid:
            raise ValueError("valid must be specified.")
        train_indices = dataset.get_idx_for_split(train)
        valid_indices = dataset.get_idx_for_split(valid)
        if not test:
            test = valid
        test_indices = dataset.get_idx_for_split(test)
        inner_cv = ValidationSplit(
            np.arange(len(train_indices), len(train_indices) + len(valid_indices))
        )

    clf_lib, clf_type = clf_type.split("/")
    if clf_lib == "sk":
        from sklearn.model_selection import GridSearchCV

        from ertk.sklearn.models import get_sk_model

        clf = get_sk_model(clf_type, **model_args)
        if train:
            # A hack so that the ValidationSplit works properly.
            train_indices = np.concatenate([train_indices, valid_indices])
            grid_search = GridSearchVal
        else:
            grid_search = GridSearchCV
        clf = grid_search(
            Pipeline([("transform", transformer), ("clf", clf)]),
            {f"clf__{k}": v for k, v in param_grid.items()},
            scoring="balanced_accuracy",
            cv=inner_cv,
            verbose=max(verbose, 0),
            n_jobs=1 if use_inner_cv and not train else n_jobs,
        )
        if inner_part:
            fit_params["groups"] = dataset.get_group_indices(inner_part)
        if not use_inner_cv:
            # Get best params then pass best to main cross-validation routine
            # XXX: This exists because of mistake in our original implementation.
            clf.fit(
                dataset.x,
                dataset.labels,
                groups=dataset.get_group_indices(cv_part) if cv_part else None,
                clf__sample_weight=sample_weight,
            )
            params = clf.best_params_
            clf = clf.best_estimator_
    elif clf_lib == "tf":
        from keras.callbacks import TensorBoard
        from keras.optimizers import get as get_optimizer

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
        # No parallel CV to avoid running out of GPU memory
        n_jobs = 1
        fit_params.update(
            {
                "epochs": epochs,
                "callbacks": callbacks,
                "batch_size": batch_size,
                "data_fn": None,
                "n_gpus": n_gpus,
                "log_dir": logdir,
            }
        )
        model_args.update(
            {"n_features": dataset.n_features, "n_classes": dataset.n_classes}
        )

        def model_fn():
            model = get_tf_model(clf_type, **model_args)
            model.compile(
                get_optimizer("adam")(learning_rate=learning_rate),
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
        from omegaconf import OmegaConf

        from ertk.pytorch.models import get_pt_model
        from ertk.pytorch.train import PyTorchLoggingConfig, PyTorchTrainConfig

        n_jobs = 1

        train_config = PyTorchTrainConfig(
            batch_size=batch_size,
            n_gpus=n_gpus,
            epochs=epochs,
            logging=PyTorchLoggingConfig(log_dir=str(logdir)),
        )
        fit_params.update({"train_config": train_config})

        # model_config = PyTorchModelConfig(
        #     optimiser="adam",
        #     learning_rate=learning_rate,
        #     n_features=dataset.n_features,
        #     n_classes=dataset.n_classes,
        #     loss="cross_entropy",
        # )
        model_config = OmegaConf.create(model_args)
        model_config.n_features = dataset.n_features
        model_config.n_classes = dataset.n_classes

        def model_fn():
            model = get_pt_model(clf_type, config=model_config)
            return model if transformer is None else (transformer, model)

        params = {**model_config, **PyTorchTrainConfig.to_dictconfig(train_config)}
        clf = model_fn
    else:
        raise ValueError(f"Invalid classifier type {clf_type}")

    if verbose > -1:
        print("== Dataset ==")
        print(dataset)
        print("== Training/evaluation settings ==")
        print(f"Using {dataset.n_classes}-class classification.")
        print(f"Using {n_jobs} parallel jobs.")
        print(f"Using classifier {clf}.")
        print(f"Using fit_params={fit_params}")
        if train:
            print(f"Training set: {train}")
            print(f"Validation set: {valid}")
            print(f"Test set: {test}")
        else:
            print(f"Using {kfold} k-fold CV.")
            if cv_part:
                print(f"Using {cv_part} as split partition.")
            if use_inner_cv:
                if inner_part:
                    print(f"Using {inner_part} as inner split partition.")
                print(f"Using {inner_cv} as inner CV splitter.")
        if balanced:
            print("Using balanced sample weights.")
        if normalise == "online":
            print("Using 'online' normalisation.")
        elif normalise != "none":
            print(f"Normalising globally using {normalise} partition.")

    dfs = []
    for rep in range(1, reps + 1):
        if verbose > -1:
            print(f"Rep {rep}/{reps}")
        if train:
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
        else:
            df = dataset_cross_validation(
                clf,
                dataset,
                clf_lib=clf_lib,
                partition=cv_part,
                label=label,
                cv=cv,
                verbose=max(verbose, 0),
                n_jobs=n_jobs,
                fit_params=fit_params,
            )
        if "params" not in df:
            df["params"] = [json.dumps(params, default=str)] * len(df)
        dfs.append(df)
    df = pd.concat(dfs, keys=list(range(1, reps + 1)), names=["rep"])
    df["clf"] = f"{clf_lib}/{clf_type}"
    df["features"] = features
    df["corpus"] = dataset.corpus
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
