import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from emotion_recognition.classification import PrecomputedSVC
from emotion_recognition.dataset import LabelledDataset, NetCDFDataset
from emotion_recognition.tensorflow.models.aldeneh2017 import \
    full_model as aldeneh_model
from emotion_recognition.tensorflow.utils import create_tf_dataset_ragged
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import (SCORERS, accuracy_score, balanced_accuracy_score,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     LeaveOneGroupOut, cross_validate)
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


# Fully connected feedforward networks.
def basic_keras_model(n_features: int, n_classes: int) -> Model:
    """Creates a Keras model with a hidden layer and sigmoid activation,
    without dropout.
    """
    inputs = Input((n_features,), name='input')
    dense_1 = Dense(512, activation='sigmoid')(inputs)
    x = Dense(n_classes, activation='softmax')(dense_1)
    return Model(inputs=inputs, outputs=x)


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


def get_vec_model(kind='1layer', n_features=None, n_classes=None):
    if kind == 'basic':
        model = basic_keras_model(n_features, n_classes)
    elif kind == '1layer':
        model = dense_keras_model(n_features, n_classes, layers=1)
    elif kind == '2layer':
        model = dense_keras_model(n_features, n_classes, layers=2)
    elif kind == '3layer':
        model = dense_keras_model(n_features, n_classes, layers=3)
    else:
        raise NotImplementedError("Other kinds of dense model are not "
                                  "currently implemented.")
    model.compile(
        optimizer=Adam(learning_rate=0.0001), metrics=['categorical_accuracy'],
        loss='categorical_crossentropy'
    )
    return model


# Sequence models
def get_seq_model(kind='aldeneh', n_features=None, n_classes=None):
    if kind == 'aldeneh':
        model = aldeneh_model(n_features, n_classes)
    else:
        raise NotImplementedError("Other kinds of convolutional model are not "
                                  "currently implemented.")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        metrics=['sparse_categorical_accuracy'],
        loss='sparse_categorical_crossentropy'
    )
    return model


def test_classifier(type_: str,
                    kind: str,
                    dataset: LabelledDataset,
                    reps: int = 1,
                    results: Optional[Union[str, Path]] = None):
    splitter = LeaveOneGroupOut()
    if dataset.n_speakers > 12:
        splitter = GroupKFold(6)

    class_counts = np.bincount(dataset.y.astype(int))
    class_weight = dataset.n_instances / (dataset.n_classes * class_counts)
    # Necessary until scikeras supports passing in class_weights directly
    sample_weight = class_weight[dataset.y.astype(int)]

    metrics = (['uar', 'war'] + [x + '_rec' for x in dataset.classes]
               + [x + '_prec' for x in dataset.classes])
    df = pd.DataFrame(index=pd.RangeIndex(reps, name='rep'),
                      columns=metrics + ['params'])

    for rep in range(reps):
        print("Rep {}".format(rep))
        if type_ in ['svm', 'dnn']:
            if type_ == 'svm':
                param_grid = get_svm_params(kind)
                clf = GridSearchCV(PrecomputedSVC(), param_grid, cv=splitter,
                                   scoring='balanced_accuracy', n_jobs=-1)
                clf.fit(dataset.x, dataset.y, groups=dataset.speaker_indices,
                        sample_weight=sample_weight)
                params = clf.best_params_
                clf = clf.best_estimator_
            else:
                # Force CPU only to do in parallel, supress TF errors
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                params = dict(batch_size=64, epochs=50)
                clf = KerasClassifier(
                    get_vec_model, kind=kind, n_features=dataset.n_features,
                    n_classes=dataset.n_classes, **params, verbose=False
                )
            # Dict of scoring functions to use
            scoring = {'war': SCORERS['accuracy'],
                       'uar': SCORERS['balanced_accuracy']}
            for i, c in enumerate(dataset.classes):
                scoring.update({
                    c + '_rec': make_scorer(recall_score, average=None,
                                            labels=[i]),
                    c + '_prec': make_scorer(precision_score, average=None,
                                             labels=[i])
                })

            scores = cross_validate(
                clf, dataset.x, dataset.y, cv=splitter, scoring=scoring,
                groups=dataset.speaker_group_indices,
                fit_params=dict(sample_weight=sample_weight),
                n_jobs=-1
            )
        else:  # type_ == 'cnn'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
            scores = defaultdict(list)
            params = dict(batch_size=64, epochs=100)
            for train, test in splitter.split(dataset.x, dataset.y,
                                              dataset.speaker_group_indices):
                tf.keras.backend.clear_session()

                x_train = dataset.x[train]
                y_train = dataset.y[train]
                x_test = dataset.x[test]
                y_test = dataset.y[test]
                train_data = create_tf_dataset_ragged(
                    x_train, y_train, batch_size=64, shuffle=True)
                test_data = create_tf_dataset_ragged(
                    x_test, y_test, batch_size=64, shuffle=False)

                clf = get_seq_model(kind=kind, n_features=dataset.n_features,
                                    n_classes=dataset.n_classes)
                # Use validation data just for info
                clf.fit(train_data, validation_data=test_data, epochs=50,
                        verbose=False)
                y_true = np.concatenate([x[1] for x in test_data])
                y_pred = np.argmax(clf.predict(test_data), axis=-1)

                # Scoring
                scores['test_war'].append(accuracy_score(y_true, y_pred))
                scores['test_uar'].append(balanced_accuracy_score(y_true,
                                                                  y_pred))
                for i, c in enumerate(dataset.classes):
                    scores['test_' + c + '_rec'].append(recall_score(
                        y_true, y_pred, average=None, labels=[i]))
                    scores['test_' + c + '_prec'].append(precision_score(
                        y_true, y_pred, average=None, labels=[i]))
        # Make string to add to final dataframe
        params = json.dumps(params)

        mean_scores = {k[5:]: np.mean(v) for k, v in scores.items()
                       if k.startswith('test_')}
        war = mean_scores['war']
        uar = mean_scores['uar']
        recall = tuple(mean_scores[c + '_rec'] for c in dataset.classes)
        precision = tuple(mean_scores[c + '_prec'] for c in dataset.classes)

        df.loc[rep, 'params'] = params
        df.loc[rep, 'war'] = war
        df.loc[rep, 'uar'] = uar
        for i, c in enumerate(dataset.classes):
            df.loc[rep, c + '_rec'] = recall[i]
            df.loc[rep, c + '_prec'] = precision[i]

    if results:
        results.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results)
        print("Wrote CSV to {}.".format(results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf', type=str, required=True,
                        help="The type of classifier: {svm, dnn, cnn}.")
    parser.add_argument('--data', type=Path, required=True,
                        help="The data to use.")
    parser.add_argument('--kind', type=str, required=True,
                        help="The kind of classifier.")

    parser.add_argument('--datatype', type=str, default='utterance',
                        help="The type of data: {frame, utterance}.")
    parser.add_argument('--name', type=str, help="The results output name.")
    parser.add_argument('--results', type=Path, help="Results directory.")
    parser.add_argument('--reps', type=int, default=1,
                        help="The number of repetitions to do per test.")
    args = parser.parse_args()

    if args.clf not in ['svm', 'dnn', 'cnn']:
        raise ValueError("--clf must be one of {svm, dnn, cnn}.")

    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset = NetCDFDataset(args.data)
    dataset.normalise(normaliser=StandardScaler(), scheme='speaker')
    if args.datatype == 'frame':
        dataset.pad_arrays(64)

    test_classifier(args.clf, args.kind, dataset, reps=args.reps,
                    results=args.results)


if __name__ == "__main__":
    main()
