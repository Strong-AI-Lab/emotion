import argparse
import json
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from emorec.dataset import LabelledDataset
from emorec.sklearn.models import PrecomputedSVC
from emorec.tensorflow.classification import DummyEstimator, tf_cross_validate
from emorec.tensorflow.models import (aldeneh2017_model, latif2019_model,
                                      zhang2019_model, zhao2019_model)
from emorec.tensorflow.models.zhang2019 import create_windowed_dataset
from emorec.tensorflow.utils import create_tf_dataset_ragged
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (get_scorer, make_scorer, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


# SVM classifiers
def get_svm_params(kind='linear') -> Dict[str, List]:
    param_grid = {'C': 2.0**np.arange(-6, 7, 2)}
    if kind == 'linear':
        param_grid.update({'kernel': ['poly'], 'degree': [1], 'coef0': [0]})
    elif kind == 'poly2':
        param_grid.update({'kernel': ['poly'], 'degree': [2]})
        param_grid['coef0'] = [-1, 0, 1]
    elif kind == 'poly3':
        param_grid.update({'kernel': ['poly'], 'degree': [3]})
        param_grid['coef0'] = [-1, 0, 1]
    elif kind == 'rbf':
        param_grid['kernel'] = ['rbf']
        param_grid['gamma'] = 2.0**np.arange(-12, -1, 2)
    else:
        raise NotImplementedError("Other kinds of SVM are not currently "
                                  "implemented.")
    return param_grid


# Random forest classifiers
def get_rf_params() -> Dict[str, List]:
    param_grid = {'n_estimators': [100, 250, 500],
                  'max_depth': [None, 10, 20, 50]}
    return param_grid


# Fully connected feedforward networks.
def dense_keras_model(n_features: int, n_classes: int,
                      layers: int = 1) -> Model:
    """Creates a Keras model with hidden layers and ReLU activation,
    with 50% dropout.
    """
    inputs = Input((n_features,), name='input')
    x = inputs
    for _ in range(layers):
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)


def get_vec_model(kind: str = '1layer', n_features: int = None,
                  n_classes: int = None, lr: float = 0.0001) -> Model:
    if kind == '1layer':
        model = dense_keras_model(n_features, n_classes, layers=1)
    elif kind == '2layer':
        model = dense_keras_model(n_features, n_classes, layers=2)
    elif kind == '3layer':
        model = dense_keras_model(n_features, n_classes, layers=3)
    else:
        raise NotImplementedError("Other kinds of dense model are not "
                                  "currently implemented.")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        metrics=['sparse_categorical_crossentropy'],
        loss='sparse_categorical_crossentropy'
    )
    return model


# Sequence models
def get_seq_model(kind: str = 'aldeneh2017', n_features: int = None,
                  n_classes: int = None, lr: float = 0.0001) -> Model:
    if kind == 'aldeneh2017':
        model = aldeneh2017_model(n_features, n_classes)
    elif kind == 'latif2019':
        model = latif2019_model(n_classes)
    elif kind == 'zhang2019':
        model = zhang2019_model(n_classes)
    elif kind == 'zhao2019':
        model = zhao2019_model(n_features, n_classes)
    else:
        raise NotImplementedError("Other kinds of convolutional model are not "
                                  "currently implemented.")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        metrics=['sparse_categorical_accuracy'],
        loss='sparse_categorical_crossentropy'
    )
    return model


def test_classifier(kind: str,
                    train_data: LabelledDataset,
                    test_data: LabelledDataset,
                    reps: int = 1,
                    results: Optional[Path] = None,
                    logs: Optional[Path] = None,
                    verbose: bool = False,
                    lr: float = 1e-4,
                    epochs: int = 50,
                    bs: int = 64):
    class_weight = (train_data.n_instances
                    / (train_data.n_classes * train_data.class_counts))
    # Necessary until scikeras supports passing in class_weights directly
    sample_weight = class_weight[train_data.y]

    metrics = (['uar', 'war'] + [x + '_rec' for x in train_data.classes]
               + [x + '_prec' for x in train_data.classes])
    df = pd.DataFrame(index=pd.RangeIndex(1, reps + 1, name='rep'),
                      columns=metrics + ['params'])
    scoring = {'war': get_scorer('accuracy'),
               'uar': get_scorer('balanced_accuracy')}
    for i, c in enumerate(train_data.classes):
        scoring.update({
            c + '_rec': make_scorer(recall_score, average=None, labels=[i]),
            c + '_prec': make_scorer(precision_score, average=None, labels=[i])
        })

    type_ = ''
    _slash = kind.find('/')
    if kind.find('/') >= 0:
        type_ = kind[:_slash]
        kind = kind[_slash + 1:]
    for rep in range(1, reps + 1):
        print(f"Rep {rep}/{reps}")
        if type_ in ['svm', 'mlp'] or kind == 'rf':
            if type_ == 'mlp':
                # Force CPU only to do in parallel, supress TF errors
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                params = dict(lr=lr, batch_size=bs, epochs=epochs)
                clf = KerasClassifier(
                    get_vec_model, kind=kind, n_features=train_data.n_features,
                    n_classes=train_data.n_classes, **params, verbose=False
                )
            else:
                if type_ == 'svm':
                    param_grid = get_svm_params(kind)
                    _clf = PrecomputedSVC()
                else:
                    param_grid = get_rf_params()
                    _clf = RandomForestClassifier()
                # Inner CV for hyperparameter optimisation
                cv = GroupKFold(5)
                if len(set(train_data.speaker_group_indices)) < 5:
                    cv = LeaveOneGroupOut()
                clf = GridSearchCV(_clf, param_grid, cv=cv,
                                   scoring='balanced_accuracy', n_jobs=-1)
                # Get best hyperparameters through inner CV
                clf.fit(
                    train_data.x, train_data.y,
                    groups=train_data.speaker_group_indices,
                    sample_weight=sample_weight
                )
                params = clf.best_params_
                clf = clf.best_estimator_
            clf.fit(train_data.x, train_data.y, sample_weight=sample_weight)
            y_pred = clf.predict(test_data.x)
            dummy = DummyEstimator(y_pred)
            scores = defaultdict(list)
            _scores = _score(dummy, y_pred, test_data.y, scoring)
            for k, v in _scores.items():
                scores['test_' + k].append(v)
        else:  # type_ == 'cnn'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
            data_fn = create_tf_dataset_ragged
            if kind == 'zhang2019':
                data_fn = create_windowed_dataset
            data_fn = partial(data_fn, batch_size=bs)
            params = dict(lr=lr, batch_size=bs, epochs=epochs)

            # To print model params
            _model = get_seq_model(
                kind=kind, n_features=train_data.n_features,
                n_classes=train_data.n_classes, lr=lr
            )
            _model.summary()
            del _model
            tf.keras.backend.clear_session()

            model_fn = partial(
                get_seq_model, kind=kind, n_features=train_data.n_features,
                n_classes=train_data.n_classes, lr=lr
            )
            scores = tf_cross_validate(
                model_fn, train_data.x, train_data.y, cv=splitter, scoring=scoring,
                groups=train_data.speaker_group_indices, data_fn=data_fn,
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
        war = mean_scores['war']
        uar = mean_scores['uar']
        recall = tuple(mean_scores[c + '_rec'] for c in train_data.classes)
        precision = tuple(mean_scores[c + '_prec'] for c in train_data.classes)

        df.loc[rep, 'params'] = params
        df.loc[rep, 'war'] = war
        df.loc[rep, 'uar'] = uar
        for i, c in enumerate(train_data.classes):
            df.loc[rep, c + '_rec'] = recall[i]
            df.loc[rep, c + '_prec'] = precision[i]

    if results:
        results.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results)
        print(f"Wrote CSV to {results}")
    else:
        print(df.to_string())


def main():
    parser = argparse.ArgumentParser()

    # Required options
    parser.add_argument('--kind', type=str, required=True,
                        help="The kind of classifier.")
    parser.add_argument('--train', type=Path, required=True,
                        help="The train data.")
    parser.add_argument('--test', type=Path, required=True,
                        help="The test data.")

    # Dataset options
    parser.add_argument('--pad', type=int,
                        help="Pad input sequences to this length.")
    parser.add_argument(
        '--clip', type=int,
        help="Clips input sequences to this maximum length, after any padding."
    )

    # Results options
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

    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    train_data = LabelledDataset(args.train)
    test_data = LabelledDataset(args.test)
    # helplessness is for SmartKom
    train_data.map_classes({'helplessness': 'sadness'})
    test_data.map_classes({'helplessness': 'sadness'})
    train_data.remove_classes(keep=['anger', 'happiness', 'sadness'])
    test_data.remove_classes(keep=['anger', 'happiness', 'sadness'])
    train_data.normalise(normaliser=StandardScaler(), scheme='speaker')
    test_data.normalise(normaliser=StandardScaler(), scheme='speaker')
    if args.pad:
        train_data.pad_arrays(args.pad)
        test_data.pad_arrays(args.pad)
    if args.clip:
        train_data.clip_arrays(args.clip)
        test_data.clip_arrays(args.clip)

    test_classifier(
        args.kind, train_data, test_data, reps=args.reps, results=args.results,
        logs=args.logs, verbose=args.verbose, lr=args.learning_rate,
        epochs=args.epochs, bs=args.batch_size
    )


if __name__ == "__main__":
    main()
