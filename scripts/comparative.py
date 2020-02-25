import os
import traceback
from functools import partial

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
    output_dir = os.path.join(RESULTS_DIR, dataset.corpus, 'svm', kernel)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, '{}.csv'.format(config)))


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
                                              patience=5, mode='max'),
            keras.callbacks.TensorBoard(
                os.path.join('logs', dataset.corpus, 'dnn', kind, config,
                             'foldN'),
                write_graph=False
            )
        ],
        class_weight=class_weight,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        verbose=False
    )
    print_results(df)
    output_dir = os.path.join(RESULTS_DIR, dataset.corpus, 'dnn', kind)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, '{}.csv'.format(config)))


def main():
    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    for corpus in CLASSIFICATION_CORPORA:
        for config in CLASSIFICATION_CONFIGS:
            try:
                print(corpus, config)
                dataset = UtteranceDataset(
                    '{}/output/{}.arff'.format(corpus, config),
                    normaliser=StandardScaler(),
                    normalise_method='speaker'
                )
                print()
                test_svm_classifier(dataset, config, kernel='linear')
                test_svm_classifier(dataset, config, kernel='poly2')
                test_svm_classifier(dataset, config, kernel='poly3')
                test_svm_classifier(dataset, config, kernel='rbf')
                test_dense_model(dataset, config, kind='basic')
                test_dense_model(dataset, config, kind='1layer')
                test_dense_model(dataset, config, kind='2layer')
                test_dense_model(dataset, config, kind='3layer')
            except Exception:
                print("Failed:", corpus, config)
                traceback.print_exc()

    for corpus in CLASSIFICATION_CORPORA:
        try:
            print(corpus, 'audeep')
            dataset = NetCDFDataset(
                '{}/output/audeep.nc'.format(corpus), corpus,
                normaliser=StandardScaler(), normalise_method='speaker'
            )
            print()
            test_svm_classifier(dataset, 'audeep', kernel='linear')
            test_svm_classifier(dataset, 'audeep', kernel='poly2')
            test_svm_classifier(dataset, 'audeep', kernel='poly3')
            test_svm_classifier(dataset, 'audeep', kernel='rbf')
            test_dense_model(dataset, 'audeep', kind='basic')
            test_dense_model(dataset, 'audeep', kind='1layer')
            test_dense_model(dataset, 'audeep', kind='2layer')
            test_dense_model(dataset, 'audeep', kind='3layer')
        except Exception:
            print("Failed:", corpus, 'audeep')
            traceback.print_exc()


if __name__ == "__main__":
    main()
