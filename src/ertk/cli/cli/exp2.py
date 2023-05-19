import json
import logging
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ertk.classification import (
    dataset_cross_validation,
    dataset_train_val_test,
    get_balanced_sample_weights,
)
from ertk.cli._utils import debug_args
from ertk.config import get_arg_mapping
from ertk.dataset import load_datasets_config
from ertk.sklearn.utils import GridSearchVal
from ertk.train import ExperimentConfig, ValidationSplit, get_cv_splitter
from ertk.transform import SequenceTransformWrapper


@click.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("restargs", type=str, nargs=-1)
@debug_args
def main(config_path: Path, restargs: Tuple[str], verbose: int):
    config = ExperimentConfig.from_file(config_path, override=list(restargs))

    np.set_printoptions(precision=4, threshold=10)
    if verbose > 0:
        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)
        np.set_printoptions()
        logging.info(f"Config: {ExperimentConfig.to_string(config)}")

    dataset = load_datasets_config(config.data)

    transform = config.training.transform.name
    transformer = {"std": StandardScaler, "minmax": MinMaxScaler}[transform]()
    normalise = config.training.normalise
    if normalise == "none":
        transformer = None
    elif normalise == "online" and len(dataset.x[0].shape) > 1:
        transformer = SequenceTransformWrapper(
            transformer, config.training.seq_transform
        )
    elif normalise != "online":
        dataset.normalise(normaliser=transformer, partition=normalise)
        transformer = None

    fit_params = {}

    sample_weight = None
    balanced = config.training.balanced
    if balanced:
        sample_weight = get_balanced_sample_weights(dataset.y)

    param_grid = config.model.param_grid
    if config.model.param_grid_path:
        param_grid.update(get_arg_mapping(config.model.param_grid_path))

    evaluation = config.eval
    if evaluation is None:
        raise ValueError("Must specify evaluation to run")
    cv_conf = evaluation.cv

    if evaluation.inner_kfold is not None:
        inner_cv = get_cv_splitter(
            bool(evaluation.inner_part),
            evaluation.inner_kfold,
            shuffle=True,
            random_state=54321,
        )
    if cv_conf is not None:
        fit_params["sample_weight"] = sample_weight
        cv = get_cv_splitter(
            bool(cv_conf.part),
            cv_conf.kfold,
            test_size=cv_conf.test_size,
        )
        if evaluation.inner_kfold is None and not param_grid:
            raise ValueError(
                "Need to specify inner_kfold when doing CV with param_grid"
            )
    elif evaluation.train:
        if not evaluation.valid:
            raise ValueError("valid must be specified.")
        if not evaluation.test:
            evaluation.test = evaluation.valid
        train_indices = dataset.get_idx_for_split(evaluation.train)
        valid_indices = dataset.get_idx_for_split(evaluation.valid)
        test_indices = dataset.get_idx_for_split(evaluation.test)
    else:
        raise ValueError("Either eval.cv or eval.train must be set.")
    n_jobs = config.training.n_jobs

    clf_type = config.model.type
    clf_lib, clf_type = clf_type.split("/")
    if clf_lib == "sk":
        from sklearn.model_selection import GridSearchCV

        from ertk.sklearn.models import get_sk_model

        if evaluation.inner_kfold is not None and cv_conf is not None:
            # Outer and inner cross-validation
            inner_n_jobs = 1
        else:
            # Only inner cross-validation
            inner_n_jobs = n_jobs

        clf = get_sk_model(clf_type, **config.model.config)
        if evaluation.train and evaluation.inner_kfold is None:
            # Predefinted train/val split
            # This is a hack so that the ValidationSplit works properly.
            inner_cv = ValidationSplit(
                np.arange(len(train_indices)),
                np.arange(len(train_indices), len(train_indices) + len(valid_indices)),
            )
            train_indices = np.concatenate([train_indices, valid_indices])
            grid_search = GridSearchVal
        else:
            grid_search = GridSearchCV

        clf = Pipeline([("transform", transformer), ("clf", clf)])
        if "sample_weight" in fit_params:
            fit_params["clf__sample_weight"] = fit_params.pop("sample_weight")

        if param_grid:
            clf = grid_search(
                clf,
                {f"clf__{k}": v for k, v in param_grid.items()},
                scoring="balanced_accuracy",
                cv=inner_cv,
                verbose=config.training.verbose,
                n_jobs=inner_n_jobs,
            )
            if evaluation.inner_part:
                fit_params["groups"] = dataset.get_group_indices(evaluation.inner_part)
    elif clf_lib == "tf":
        from keras.callbacks import TensorBoard
        from keras.optimizers import get as get_optimizer

        from ertk.tensorflow.classification import tf_classification_metrics
        from ertk.tensorflow.models import TFModelConfig, get_tf_model
        from ertk.tensorflow.train import TFTrainConfig

        if config.training.tensorflow:
            tf_config = TFTrainConfig.from_config(config.training.tensorflow)
        else:
            tf_config = TFTrainConfig()

        callbacks = []
        if tf_config.logging.log_dir is not None:
            callbacks.append(
                TensorBoard(
                    log_dir=tf_config.logging.log_dir,
                    profile_batch=0,
                    write_graph=False,
                    write_images=False,
                )
            )
        # No parallel CV to avoid running out of GPU memory
        n_jobs = 1
        fit_params.update(
            {
                "epochs": tf_config.epochs,
                "callbacks": callbacks,
                "batch_size": tf_config.batch_size,
                "data_fn": tf_config.data_fn,
                "n_gpus": tf_config.n_gpus,
                "log_dir": tf_config.logging.log_dir,
            }
        )
        model_config = config.model.config
        model_config.n_features = dataset.n_features
        model_config.n_classes = dataset.n_classes

        def model_fn():
            model = get_tf_model(clf_type, **model_config)
            model.compile(
                get_optimizer("adam", learning_rate=model_config.learning_rate),
                loss="sparse_categorical_crossentropy",
                weighted_metrics=tf_classification_metrics(),
            )
            return Pipeline([("transform", transformer), ("clf", model)])

        params = {
            "optimiser": "adam",
            "learning_rate": model_config.learning_rate,
            "batch_size": tf_config.batch_size,
            **config.model.config,
            **fit_params,
        }
        clf = model_fn
    elif clf_lib == "pt":
        from ertk.pytorch.models import (
            ERTKPyTorchModel,
            PyTorchModelConfig,
            get_pt_model,
        )
        from ertk.pytorch.train import PyTorchTrainConfig

        pt_config = PyTorchTrainConfig.from_config(config.training.pytorch)

        n_jobs = 1
        fit_params.update({"train_config": pt_config})

        model_cls = ERTKPyTorchModel.get_model_class(clf_type)
        model_config = model_cls.get_config_type().from_config(config.model.config)
        if model_config.n_features == -1:
            model_config.n_features = dataset.n_features
        elif model_config.n_features != dataset.n_features:
            logging.warning("`n_features` differs between dataset and model_config")
        if model_config.n_classes == -1:
            model_config.n_classes = dataset.n_classes
        elif model_config.n_classes != dataset.n_classes:
            raise ValueError("`n_classes` differs between dataset and model_config")

        def model_fn():
            model = get_pt_model(clf_type, config=model_config)
            return model if transformer is None else (transformer, model)

        params = {
            **PyTorchModelConfig.to_dictconfig(model_config),
            **PyTorchTrainConfig.to_dictconfig(pt_config),
        }
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
        if evaluation.train:
            print(f"Training set: {evaluation.train} ({len(train_indices)})")
            print(f"Validation set: {evaluation.valid} ({len(valid_indices)})")
            print(f"Test set: {evaluation.test} ({len(test_indices)})")
        elif cv_conf:
            print(f"Using {cv_conf.kfold} k-fold CV.")
            if cv_conf.part:
                print(f"Using {cv_conf.part} as split partition.")
            if evaluation.inner_kfold is not None:
                if evaluation.inner_part:
                    print(f"Using {evaluation.inner_part} as inner split partition.")
                print(f"Using {evaluation.inner_kfold} as inner CV splitter.")
        if balanced:
            print("Using balanced sample weights.")
        if normalise == "online":
            print("Using 'online' normalisation.")
        elif normalise != "none":
            print(f"Normalising globally using {normalise} partition.")
        if normalise != "none":
            print(f"Using transformer {transformer}")

    dfs = []
    _preds = []
    for rep in range(config.training.reps):
        if verbose > -1:
            print(f"Rep {rep}/{config.training.reps}")
        if evaluation.train:
            res = dataset_train_val_test(
                clf,
                dataset=dataset,
                train_idx=train_indices,
                valid_idx=valid_indices,
                test_idx=test_indices,
                label=config.training.label,
                clf_lib=clf_lib,
                sample_weight=sample_weight,
                verbose=config.training.verbose,
                fit_params=fit_params,
            )
        elif cv_conf:
            res = dataset_cross_validation(
                clf,
                dataset,
                clf_lib=clf_lib,
                partition=cv_conf.part,
                label=config.training.label,
                cv=cv,
                verbose=config.training.verbose,
                n_jobs=n_jobs,
                fit_params=fit_params,
            )
        else:
            raise RuntimeError()
        df = res.scores_df
        if "params" not in df:
            df["params"] = [json.dumps(params, default=str)] * len(df)
        dfs.append(df)
        if res.predictions is not None:
            _preds.append(res.predictions)
    if _preds:
        preds = np.stack(_preds)
    else:
        preds = np.empty(0)
    df = pd.concat(dfs, keys=pd.RangeIndex(config.training.reps), names=["rep"])
    df["clf"] = f"{clf_lib}/{clf_type}"
    df["features"] = config.data.features
    df["corpus"] = dataset.corpus
    if config.results:
        results = Path(config.results)
        results.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results)
        print(f"Wrote CSV to {results}")
        if preds.size > 0:
            np.save(results.with_suffix(".preds.npy"), preds)
    else:
        print(df)
    print(df.mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
