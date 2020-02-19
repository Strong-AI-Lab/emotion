"""Implementation of the paper,

Z. Aldeneh and E. Mower Provost, 'Using regional saliency for speech emotion
recognition', in IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), 2017, pp. 2741–2745,
doi: 10.1109/ICASSP.2017.7952655
"""

import os
from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from python.classification import (PrecomputedSVC, SKLearnClassifier,
                                   TFClassifier, print_results, test_model)
from python.dataset import FrameDataset, UtteranceDataset
from python.tensorflow import BatchedFrameSequence, BatchedSequence

RESULTS_DIR = 'results/aldeneh2017'


def get_dense_model(n_features, n_classes):
    inputs = keras.layers.Input(shape=(n_features,), name='input')
    x = keras.layers.Dense(1024, activation='relu',
                           kernel_initializer='he_normal',
                           name='dense_1')(inputs)
    x = keras.layers.Dense(1024, activation='relu',
                           kernel_initializer='he_normal', name='dense_2')(x)
    x = keras.layers.Dense(n_classes, activation='softmax',
                           kernel_initializer='he_normal',
                           name='emotion_prediction')(x)
    return keras.Model(inputs=inputs, outputs=x, name='aldeneh_dense_model')


def get_conv_model(n_features, n_classes, n_filters=128, kernel_size=8):
    inputs = keras.layers.Input(shape=(None, n_features), name='input')
    x = keras.layers.Conv1D(
        n_filters, kernel_size, activation='relu',
        kernel_initializer='he_normal', name='conv'
    )(inputs)
    x = keras.layers.GlobalMaxPool1D(name='maxpool')(x)
    x = keras.layers.Dense(1024, activation='relu',
                           kernel_initializer='he_normal', name='dense_1')(x)
    x = keras.layers.Dense(1024, activation='relu',
                           kernel_initializer='he_normal', name='dense_2')(x)
    x = keras.layers.Dense(
        n_classes, activation='softmax', kernel_initializer='he_normal',
        name='emotion_prediction'
    )(x)
    return keras.Model(inputs=inputs, outputs=x, name='aldeneh_conv_model')


def get_full_model(n_features, n_classes):
    inputs = keras.layers.Input(shape=(None, n_features), name='input')
    x = keras.layers.Conv1D(384, 8, activation='relu',
                            kernel_initializer='he_normal',
                            name='conv8')(inputs)
    c1 = keras.layers.GlobalMaxPool1D(name='maxpool_1')(x)

    x = keras.layers.Conv1D(384, 16, activation='relu',
                            kernel_initializer='he_normal',
                            name='conv16')(inputs)
    c2 = keras.layers.GlobalMaxPool1D(name='maxpool_2')(x)

    x = keras.layers.Conv1D(384, 32, activation='relu',
                            kernel_initializer='he_normal',
                            name='conv32')(inputs)
    c3 = keras.layers.GlobalMaxPool1D(name='maxpool_3')(x)

    x = keras.layers.Conv1D(384, 64, activation='relu',
                            kernel_initializer='he_normal',
                            name='conv64')(inputs)
    c4 = keras.layers.GlobalMaxPool1D(name='maxpool_4')(x)

    x = keras.layers.Concatenate(name='concatenate')([c1, c2, c3, c4])
    x = keras.layers.Dense(1024, activation='relu',
                           kernel_initializer='he_normal', name='dense_1')(x)
    x = keras.layers.Dense(1024, activation='relu',
                           kernel_initializer='he_normal', name='dense_2')(x)
    x = keras.layers.Dense(n_classes, activation='softmax',
                           kernel_initializer='he_normal',
                           name='emotion_prediction')(x)
    return keras.Model(inputs=inputs, outputs=x, name='aldeneh_conv_model')


def optimizer_fn():
    return keras.optimizers.RMSprop(learning_rate=0.0001)


def callbacks_fn():
    return [
        keras.callbacks.EarlyStopping(monitor='val_uar', patience=10,
                                      restore_best_weights=True, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_uar', factor=1 / 1.4,
                                          patience=0, mode='max')
    ]


def test_svm_models():
    param_grid = ParameterGrid({
        'C': 2.0**np.arange(0, 13, 2),
        'gamma': 2.0**np.arange(-15, -2, 2)
    })

    for corpus in ['iemocap', 'msp-improv']:
        for config in ['IS09_emotion', 'IS13_IS09_func', 'GeMAPSv01a',
                       'eGeMAPSv01a']:
            print(config)
            dataset = UtteranceDataset(
                '{}/output/{}.arff'.format(corpus, config),
                normaliser=StandardScaler(),
                normalise_method='speaker'
            )
            print()

            df = test_model(
                SKLearnClassifier(partial(PrecomputedSVC, kernel='rbf',
                                          class_weight='balanced')),
                dataset,
                reps=1,
                splitter=LeaveOneGroupOut(),
                param_grid=param_grid,
                cv_score_fn=partial(recall_score, average='macro')
            )

            print_results(df)
            df.to_csv(os.path.join(RESULTS_DIR, corpus,
                                   '{}.csv'.format(config)))


def test_dense_model():
    model = get_dense_model(480, 4)
    model.summary()
    keras.backend.clear_session()
    del model

    for corpus in ['iemocap', 'msp-improv']:
        print("logmel_IS09_func")
        dataset = UtteranceDataset('{}/output/logmel_IS09_func.arff'.format(
            corpus), normaliser=StandardScaler(), normalise_method='speaker')
        print()

        class_weight = dataset.n_instances / \
            (dataset.n_classes * np.bincount(dataset.y.astype(np.int)))
        class_weight = dict(zip(range(dataset.n_classes), class_weight))

        df = test_model(
            TFClassifier(partial(get_dense_model, dataset.n_features,
                                 dataset.n_classes)),
            dataset,
            reps=1,
            splitter=LeaveOneGroupOut(),
            n_epochs=50,
            class_weight=class_weight,
            data_fn=partial(BatchedSequence, batch_size=50),
            callbacks=callbacks_fn(),
            optimizer=optimizer_fn(),
            verbose=False
        )

        print_results(df)
        df.to_csv(os.path.join(RESULTS_DIR, corpus, 'logmel_func.csv'))


def test_conv_models():
    model = get_conv_model(40, 4)
    model.summary()
    keras.backend.clear_session()
    del model

    for corpus in ['iemocap', 'msp-improv']:
        print("")
        dataset = FrameDataset('{}/output/logmel.arff.bin'.format(corpus),
                               normaliser=StandardScaler(),
                               normalise_method='speaker')
        dataset.pad_arrays(32)
        print()

        class_weight = dataset.n_instances / \
            (dataset.n_classes * np.bincount(dataset.y.astype(np.int)))
        class_weight = dict(zip(range(dataset.n_classes), class_weight))

        for n_filters, kernel_size in [(384, 8), (288, 16), (208, 32),
                                       (128, 64), (80, 128)]:
            print("(n_filters, kernel_size) = ({}, {})".format(n_filters,
                                                               kernel_size))
            df = test_model(
                TFClassifier(partial(
                    get_conv_model, dataset.n_features, dataset.n_classes,
                    n_filters=n_filters, kernel_size=kernel_size
                )),
                dataset,
                reps=1,
                splitter=LeaveOneGroupOut(),
                n_epochs=50,
                class_weight=class_weight,
                data_fn=partial(BatchedFrameSequence, batch_size=50),
                callbacks=callbacks_fn(),
                optimizer=optimizer_fn(),
                verbose=False
            )

            print_results(df)
            df.to_csv(os.path.join(RESULTS_DIR, corpus,
                                   'logmel_{}.csv'.format(kernel_size)))


def test_full_model():
    model = get_full_model(40, 4)
    model.summary()
    keras.backend.clear_session()
    del model

    for corpus in ['iemocap', 'msp-improv']:
        dataset = FrameDataset('{}/output/logmel.arff.bin'.format(corpus),
                               normaliser=StandardScaler(),
                               normalise_method='speaker')
        dataset.pad_arrays(32)
        print()

        class_weight = dataset.n_instances / \
            (dataset.n_classes * np.bincount(dataset.y.astype(np.int)))
        class_weight = dict(zip(range(dataset.n_classes), class_weight))

        df = test_model(
            TFClassifier(partial(get_full_model, dataset.n_features,
                                 dataset.n_classes)),
            dataset,
            reps=1,
            splitter=LeaveOneGroupOut(),
            n_epochs=50,
            class_weight=class_weight,
            data_fn=BatchedFrameSequence,
            callbacks=callbacks_fn(),
            optimizer=optimizer_fn(),
            verbose=False
        )

        print_results(df)
        df.to_csv(os.path.join(RESULTS_DIR, corpus, 'logmel_full.csv'))


def main():
    os.makedirs(os.path.join(RESULTS_DIR, 'iemocap'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'msp-improv'), exist_ok=True)

    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    test_svm_models()
    test_dense_model()
    test_conv_models()
    test_full_model()


if __name__ == "__main__":
    main()
