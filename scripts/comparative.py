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
from python.dataset import FrameDataset, NetCDFDataset, UtteranceDataset

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


def get_aldeneh_full_model(n_features, n_classes):
    inputs = keras.layers.Input(shape=(None, n_features), name='input')
    x = keras.layers.Conv1D(
        384, 8, activation='relu', kernel_initializer='he_normal',
        name='conv8'
    )(inputs)
    c1 = keras.layers.GlobalMaxPool1D(name='maxpool_1')(x)

    x = keras.layers.Conv1D(
        384, 16, activation='relu', kernel_initializer='he_normal',
        name='conv16'
    )(inputs)
    c2 = keras.layers.GlobalMaxPool1D(name='maxpool_2')(x)

    x = keras.layers.Conv1D(
        384, 32, activation='relu', kernel_initializer='he_normal',
        name='conv32'
    )(inputs)
    c3 = keras.layers.GlobalMaxPool1D(name='maxpool_3')(x)

    x = keras.layers.Conv1D(
        384, 64, activation='relu', kernel_initializer='he_normal',
        name='conv64'
    )(inputs)
    c4 = keras.layers.GlobalMaxPool1D(name='maxpool_4')(x)

    x = keras.layers.Concatenate(name='concatenate')([c1, c2, c3, c4])
    x = keras.layers.Dense(1024, activation='relu',
                           kernel_initializer='he_normal', name='dense_1')(x)
    x = keras.layers.Dense(1024, activation='relu',
                           kernel_initializer='he_normal', name='dense_2')(x)
    x = keras.layers.Dense(n_classes, activation='softmax',
                           kernel_initializer='he_normal',
                           name='emotion_prediction')(x)
    return keras.Model(inputs=inputs, outputs=x, name='aldeneh_full_model')


def get_tf_dataset(x, y, batch_size=32, shuffle=True):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        data = data.shuffle(10000, reshuffle_each_iteration=True)
    return data.batch(batch_size).prefetch(8)


def get_tf_dataset_ragged(x, y, batch_size=50, shuffle=True):
    def ragged_to_dense(x, y):
        return x.to_tensor(), y

    # Sort according to length
    perm = np.argsort([len(a) for a in x])
    x = x[perm]
    y = y[perm]

    ragged = tf.RaggedTensor.from_row_lengths(np.concatenate(list(x)),
                                              [len(a) for a in x])
    data = tf.data.Dataset.from_tensor_slices((ragged, y))
    # Group similar lengths in batches, then shuffle batches
    data = data.batch(batch_size)
    if shuffle:
        data = data.shuffle(500)
    return data.map(ragged_to_dense)


def test_svm_classifier(dataset, resultname, kind='linear'):
    param_grid = ParameterGrid({
        'C': 2.0**np.arange(-6, 7, 2)
    })

    if kind == 'rbf':
        param_grid.param_grid[0]['gamma'] = 2.0**np.arange(-12, -1, 2)
    elif kind in ['poly2', 'poly3']:
        param_grid.param_grid[0]['coef0'] = [-1, 0, 1]

    splitter = LeaveOneGroupOut()
    if dataset.n_speakers > 12:
        splitter = GroupKFold(6)

    if kind == 'linear':
        params = {'kernel': 'poly', 'degree': 1, 'coef0': 0}
    elif kind == 'poly2':
        params = {'kernel': 'poly', 'degree': 2}
    elif kind == 'poly3':
        params = {'kernel': 'poly', 'degree': 3}
    elif kind == 'rbf':
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
    output_dir = Path(RESULTS_DIR) / dataset.corpus / 'svm' / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(output_dir) / '{}.csv'.format(resultname))


def test_dense_model(dataset, resultname, kind='basic'):
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
    df.to_csv(Path(output_dir) / '{}.csv'.format(resultname))


def test_conv_model(dataset, resultname, kind='aldeneh'):
    splitter = LeaveOneGroupOut()
    if dataset.n_speakers > 12:
        splitter = GroupKFold(6)

    class_weight = ((dataset.n_instances / dataset.n_classes)
                    / np.bincount(dataset.y.astype(np.int)))
    class_weight = dict(zip(range(dataset.n_classes), class_weight))

    if kind == 'aldeneh':
        model_fn = partial(get_aldeneh_full_model, dataset.n_features,
                           dataset.n_classes)

    df = test_model(
        TFClassifier(model_fn),
        dataset,
        reps=1,
        splitter=splitter,
        n_epochs=100,
        data_fn=get_tf_dataset_ragged,
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
    output_dir = Path(RESULTS_DIR) / dataset.corpus / 'cnn' / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(output_dir) / '{}.csv'.format(resultname))


parser = argparse.ArgumentParser()
parser.add_argument('--corpus', help="The corpus to test.", required=True)
parser.add_argument(
    '--clf', required=True,
    help="The type of classifier to use. One of {svm, dnn, cnn}."
)
parser.add_argument('--data', help="The data to use.", required=True)
parser.add_argument('--kind', help="The kind of classifier.", required=True)
parser.add_argument('--datatype', help="The type of data {frame, utterance}.",
                    default='utterance')
parser.add_argument('--name', help="The results output name.")


def main():
    args = parser.parse_args()

    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.clf == 'svm':
        test_fn = test_svm_classifier
    elif args.clf == 'dnn':
        test_fn = test_dense_model
    elif args.clf == 'cnn':
        test_fn = test_conv_model
        args.datatype = 'frame'

    datafile = Path(args.data)
    if datafile.suffix == '.nc':
        dataset = NetCDFDataset(
            datafile, args.corpus, normaliser=StandardScaler(),
            normalise_method='speaker'
        )
    else:
        if args.datatype == 'utterance':
            dataset = UtteranceDataset(datafile, normaliser=StandardScaler(),
                                       normalise_method='speaker')
        elif args.datatype == 'frame':
            dataset = FrameDataset(datafile, normaliser=StandardScaler(),
                                   normalise_method='speaker')
            dataset.pad_arrays(64)

    resultname = args.name or datafile.stem
    test_fn(dataset, resultname, kind=args.kind)


if __name__ == "__main__":
    main()
