import argparse
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, ParameterGrid
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from python.classification import (PrecomputedSVC, SKLearnClassifier,
                                   TFClassifier, print_results, test_model)
from python.dataset import UtteranceDataset, NetCDFDataset

RESULTS_DIR = 'results/comparative2020'
CLASSIFICATION_CORPORA = [
    'cafe', 'demos', 'emodb', 'emofilm', 'enterface', 'iemocap', 'jl',
    'msp-improv', 'portuguese', 'ravdess', 'savee', 'shemo', 'tess'
]
CLASSIFICATION_CONFIGS = ['eGeMAPSv01a', 'GeMAPSv01a', 'IS09_emotion',
                          'IS13_ComParE']
REGRESSION_CORPORA = ['iemocap', 'msp-improv', 'semaine']

parser = argparse.ArgumentParser()
parser.add_argument('corpus')
parser.add_argument('classifier')
parser.add_argument('config')
parser.add_argument('--kernel')
parser.add_argument('--kind')


def get_mlp_model(n_features, n_classes, layers=1):
    inputs = keras.Input((n_features,), name='input')
    dense_1 = keras.layers.Dense(512, activation='sigmoid')(inputs)
    x = keras.layers.Dense(n_classes, activation='softmax')(dense_1)
    return keras.Model(inputs=inputs, outputs=x)


def get_dense_model(n_features, n_classes, layers=1):
    inputs = keras.Input((n_features,), name='input')
    x = inputs
    for _ in range(layers):
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=x)


def get_tf_dataset(x, y, batch_size=32, shuffle=True):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        data = data.shuffle(10000, reshuffle_each_iteration=True)
    return data.batch(batch_size).prefetch(8)


def test_svm_classifier(dataset, config, kernel='linear'):
    param_grid = ParameterGrid({
        'C': 2.0**np.arange(-6, 7, 2)
    })

    if kernel == 'rbf':
        param_grid.param_grid[0]['gamma'] = 2.0**np.arange(-12, -1, 2)
    elif kernel in ['poly2', 'poly3']:
        param_grid.param_grid[0]['coef0'] = [-1, 0, 1]

    splitter = LeaveOneGroupOut()
    if dataset.n_speakers > 12:
        splitter = GroupKFold(6)

    if kernel == 'linear':
        params = {'kernel': 'poly', 'degree': 1, 'coef0': 0}
    elif kernel == 'poly2':
        params = {'kernel': 'poly', 'degree': 2}
    elif kernel == 'poly3':
        params = {'kernel': 'poly', 'degree': 3}
    elif kernel == 'rbf':
        params = {'kernel': 'rbf'}

    df = test_model(
        SKLearnClassifier(partial(
            PrecomputedSVC, **params, class_weight='balanced')),
        dataset,
        reps=1,
        splitter=splitter,
        param_grid=param_grid,
        cv_score_fn=partial(recall_score, average='macro')
    )

    print_results(df)
    output_dir = Path(RESULTS_DIR) / dataset.corpus / 'svm' / kernel
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(output_dir) / '{}.csv'.format(config))


def test_dense_model(dataset, config, kind='basic'):
    splitter = LeaveOneGroupOut()
    if dataset.n_speakers > 12:
        splitter = GroupKFold(6)

    class_weight = ((dataset.n_instances / dataset.n_classes)
                    / np.bincount(dataset.y.astype(np.int)))
    class_weight = dict(zip(range(dataset.n_classes), class_weight))

    if kind == 'basic':
        model_fn = partial(get_mlp_model, dataset.n_features,
                           dataset.n_classes)
    elif kind == '1layer':
        model_fn = partial(get_dense_model, dataset.n_features,
                           dataset.n_classes, layers=1)
    elif kind == '2layer':
        model_fn = partial(get_dense_model, dataset.n_features,
                           dataset.n_classes, layers=2)
    elif kind == '3layer':
        model_fn = partial(get_dense_model, dataset.n_features,
                           dataset.n_classes, layers=3)

    df = test_model(
        TFClassifier(model_fn),
        dataset,
        reps=1,
        splitter=splitter,
        n_epochs=100,
        data_fn=get_tf_dataset,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_uar', patience=20, restore_best_weights=True,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(monitor='val_uar', factor=0.5,
                                              patience=5, mode='max')
        ],
        class_weight=class_weight,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        verbose=False
    )
    print_results(df)
    output_dir = Path(RESULTS_DIR) / dataset.corpus / 'dnn' / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(output_dir) / '{}.csv'.format(config))


def main():
    args = parser.parse_args()

    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.classifier == 'svm':
        if not args.kernel:
            raise ValueError("--kernel must be passed for SVM")
        test_fn = partial(test_svm_classifier, kernel=args.kernel)
    elif args.classifier == 'dnn':
        if not args.kind:
            raise ValueError("--kind must be passed for DNN")
        test_fn = partial(test_dense_model, kind=args.kind)

    if args.config.startswith('audeep'):
        dataset = NetCDFDataset(
            '{}/output/{}.nc'.format(args.corpus, args.config), args.corpus,
            normaliser=StandardScaler(),
            normalise_method='speaker'
        )
    else:
        dataset = UtteranceDataset(
            '{}/output/{}.arff'.format(args.corpus, args.config),
            normaliser=StandardScaler(),
            normalise_method='speaker'
        )

    test_fn(dataset, args.config)


if __name__ == "__main__":
    main()
