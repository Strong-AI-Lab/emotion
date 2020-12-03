"""Runs the ALTA 2020 experiment code for testing different models."""

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from emotion_recognition.classification import PrecomputedSVC
from emotion_recognition.dataset import LabelledDataset
from emotion_recognition.tensorflow.classification import tf_cross_validate
from emotion_recognition.tensorflow.models import (aldeneh2017_model,
                                                   latif2019_model,
                                                   zhang2019_model,
                                                   zhao2019_model)
from emotion_recognition.tensorflow.models.zhang2019 import \
    create_windowed_dataset
from emotion_recognition.tensorflow.utils import create_tf_dataset_ragged
from sklearn.metrics import (get_scorer, make_scorer, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     LeaveOneGroupOut, cross_validate)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


# Sequence models
def get_seq_model(kind, n_features: int = None,
                  n_classes: int = None, lr: float = 0.0001) -> Model:
    if kind == 'aldeneh2017':
        model = aldeneh2017_model(n_features, n_classes)
    elif kind == 'latif2019':
        model = latif2019_model(n_classes)
    elif kind == 'zhang2019':
        model = zhang2019_model(n_classes)
    else:  # kind == 'zhao2019':
        model = zhao2019_model(n_features, n_classes)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        metrics=['sparse_categorical_accuracy'],
        loss='sparse_categorical_crossentropy'
    )
    return model


def test_classifier(kind: str,
                    dataset: LabelledDataset,
                    reps: int = 1,
                    results: Optional[Path] = None,
                    logs: Optional[Path] = None,
                    verbose: bool = False,
                    lr: float = 1e-4,
                    epochs: int = 50,
                    bs: int = 64):
    splitter = LeaveOneGroupOut()
    if len(dataset.speakers) > 12:
        splitter = GroupKFold(6)

    class_weight = (dataset.n_instances
                    / (dataset.n_classes * dataset.class_counts))
    # Necessary until scikeras supports passing in class_weights directly
    sample_weight = class_weight[dataset.y]

    metrics = (['uar', 'war'] + [x + '_rec' for x in dataset.classes]
               + [x + '_prec' for x in dataset.classes])
    df = pd.DataFrame(index=pd.RangeIndex(reps, name='rep'),
                      columns=metrics + ['params'])

    scoring = {'war': get_scorer('accuracy'),
               'uar': get_scorer('balanced_accuracy')}
    for i, c in enumerate(dataset.classes):
        scoring.update({
            c + '_rec': make_scorer(recall_score, average=None, labels=[i]),
            c + '_prec': make_scorer(precision_score, average=None, labels=[i])
        })

    for rep in range(reps):
        print("Rep {}".format(rep))
        if kind == 'svm':
            fit_params = dict(sample_weight=sample_weight)
            param_grid = {'C': 2.0**np.arange(-6, 7, 2), 'kernel': ['rbf'],
                          'gamma': 2.0**np.arange(-12, -1, 2)}
            clf = GridSearchCV(PrecomputedSVC(), param_grid, cv=splitter,
                               scoring='balanced_accuracy', n_jobs=-1)
            clf.fit(
                dataset.x, dataset.y, groups=dataset.speaker_group_indices,
                sample_weight=sample_weight
            )
            params = clf.best_params_
            clf = clf.best_estimator_
            scores = cross_validate(
                clf, dataset.x, dataset.y, cv=splitter, scoring=scoring,
                groups=dataset.speaker_group_indices,
                fit_params=fit_params, n_jobs=-1, verbose=int(verbose)
            )
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
            data_fn = create_tf_dataset_ragged
            if kind == 'zhang2019':
                data_fn = create_windowed_dataset
            data_fn = partial(data_fn, batch_size=bs)
            params = dict(lr=lr, batch_size=bs, epochs=epochs)

            # To print model params
            _model = get_seq_model(kind, n_features=dataset.n_features,
                                   n_classes=dataset.n_classes, lr=lr)
            _model.summary()
            del _model
            tf.keras.backend.clear_session()

            model_fn = partial(
                get_seq_model, kind, n_features=dataset.n_features,
                n_classes=dataset.n_classes, lr=lr
            )
            scores = tf_cross_validate(
                model_fn, dataset.x, dataset.y, cv=splitter, scoring=scoring,
                groups=dataset.speaker_group_indices, data_fn=data_fn,
                sample_weight=sample_weight, log_dir=None,
                fit_params=dict(epochs=epochs, verbose=verbose)
            )
            if logs:
                log_dir = logs / ('rep_' + str(rep))
                log_dir.mkdir(parents=True, exist_ok=True)
                log_df = pd.DataFrame.from_dict({
                    # e.g. (0, 'loss'): 1.431...
                    (fold, key): val
                    for fold in range(len(scores['history']))
                    for key, val in scores['history'][fold].items()
                })
                log_df.to_csv(log_dir / 'history.csv', header=True, index=True)
        # Make string to add to final dataframe
        params = json.dumps(params)

        mean_scores = {k[5:]: np.mean(v) for k, v in scores.items()
                       if k.startswith('test_')}
        df.loc[rep, 'params'] = params
        df.loc[rep, 'war'] = mean_scores['war']
        df.loc[rep, 'uar'] = mean_scores['uar']
        for c in dataset.classes:
            df.loc[rep, c + '_rec'] = mean_scores[c + '_rec']
            df.loc[rep, c + '_prec'] = mean_scores[c + '_prec']

    if results:
        results.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results)
        print("Wrote CSV to {}.".format(results))
    else:
        print(df.to_string())


def main():
    parser = argparse.ArgumentParser()

    # Required options
    parser.add_argument('--kind', type=str, required=True,
                        help="The kind of classifier.")
    parser.add_argument('--data', type=Path, required=True,
                        help="The data to use.")

    # Dataset options
    parser.add_argument('--pad', type=int,
                        help="Pad input sequences to this length.")
    parser.add_argument(
        '--clip', type=int,
        help="Clips input sequences to this maximum length, after any padding."
    )

    # Results options
    parser.add_argument('--name', type=str, help="The results output name.")
    parser.add_argument('--results', type=Path, help="Results directory.")

    # Cross-validation options
    parser.add_argument('--reps', type=int, default=1,
                        help="The number of repetitions to do per test.")

    # Misc. options
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--logs', type=Path,
                        help="Folder to write training logs per fold.")

    # Model-specific options
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    valid_models = {'svm', 'aldeneh2017', 'latif2019', 'zhang2019', 'zhao2019'}
    if args.kind not in valid_models:
        raise ValueError("--kind must be one of {}.".format(valid_models))

    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset = LabelledDataset(args.data)
    dataset.normalise(normaliser=StandardScaler(), scheme='speaker')
    if args.pad:
        dataset.pad_arrays(args.pad)
    if args.clip:
        dataset.clip_arrays(args.clip)

    test_classifier(
        args.kind, dataset, reps=args.reps, results=args.results,
        logs=args.logs, verbose=args.verbose, lr=args.learning_rate,
        epochs=args.epochs, bs=args.batch_size
    )


if __name__ == "__main__":
    main()
