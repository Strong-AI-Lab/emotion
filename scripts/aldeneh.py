"""Implementation of the paper,

Z. Aldeneh and E. Mower Provost, 'Using regional saliency for speech emotion
recognition', in IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), 2017, pp. 2741–2745,
doi: 10.1109/ICASSP.2017.7952655
"""

import os
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import recall_score
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from python.classification import (METRICS, PrecomputedSVC, cross_validate,
                                   print_results, record_metrics)
from python.dataset import FrameDataset, UtteranceDataset
from python.tensorflow import (BatchedFrameSequence, BatchedSequence,
                               tf_classification_metrics)

RESULTS_DIR = 'results/aldeneh2017'


def aldeneh_dense_model(n_features, n_classes):
    inputs = layers.Input(shape=(n_features,), name='input')
    x = layers.Dense(1024, activation='relu', kernel_initializer='he_normal',
                     name='dense_1')(inputs)
    x = layers.Dense(1024, activation='relu', kernel_initializer='he_normal',
                     name='dense_2')(x)
    x = layers.Dense(n_classes, activation='softmax',
                     kernel_initializer='he_normal',
                     name='emotion_prediction')(x)
    return keras.Model(inputs=inputs, outputs=x, name='aldeneh_dense_model')


def aldeneh_conv_model(n_features, n_classes, n_filters=128, kernel_size=8):
    inputs = layers.Input(shape=(None, n_features), name='input')
    x = layers.Conv1D(n_filters, kernel_size, activation='relu',
                      kernel_initializer='he_normal', name='conv')(inputs)
    x = layers.GlobalMaxPool1D(name='maxpool')(x)
    x = layers.Dense(1024, activation='relu', kernel_initializer='he_normal',
                     name='dense_1')(x)
    x = layers.Dense(1024, activation='relu', kernel_initializer='he_normal',
                     name='dense_2')(x)
    x = layers.Dense(n_classes, activation='softmax',
                     kernel_initializer='he_normal',
                     name='emotion_prediction')(x)
    return keras.Model(inputs=inputs, outputs=x, name='aldeneh_conv_model')


def aldeneh_full_model(n_features, n_classes):
    inputs = layers.Input(shape=(None, n_features), name='input')
    x = layers.Conv1D(384, 8, activation='relu',
                      kernel_initializer='he_normal', name='conv8')(inputs)
    c1 = layers.GlobalMaxPool1D(name='maxpool_1')(x)
    x = layers.Conv1D(384, 16, activation='relu',
                      kernel_initializer='he_normal', name='conv16')(inputs)
    c2 = layers.GlobalMaxPool1D(name='maxpool_2')(x)
    x = layers.Conv1D(384, 32, activation='relu',
                      kernel_initializer='he_normal', name='conv32')(inputs)
    c3 = layers.GlobalMaxPool1D(name='maxpool_3')(x)
    x = layers.Conv1D(384, 64, activation='relu',
                      kernel_initializer='he_normal', name='conv64')(inputs)
    c4 = layers.GlobalMaxPool1D(name='maxpool_4')(x)
    x = layers.Concatenate(name='concatenate')([c1, c2, c3, c4])
    x = layers.Dense(1024, activation='relu', kernel_initializer='he_normal',
                     name='dense_1')(x)
    x = layers.Dense(1024, activation='relu', kernel_initializer='he_normal',
                     name='dense_2')(x)
    x = layers.Dense(n_classes, activation='softmax',
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
    splitter = LeaveOneGroupOut()
    reps = 1
    param_grid = ParameterGrid(
        {'C': 2.0**np.arange(0, 13, 2), 'gamma': 2.0**np.arange(-15, -2, 2)})
    classifier_fn = partial(PrecomputedSVC, kernel='rbf',
                            class_weight='balanced')
    score_fn = partial(recall_score, average='macro')

    for corpus in ['iemocap', 'msp-improv']:
        for config in ['IS09_emotion', 'IS13_IS09_func', 'GeMAPSv01a', 'eGeMAPSv01a']:
            print(config)
            dataset = UtteranceDataset(
                '{}/output/{}.arff'.format(corpus, config),
                normaliser=StandardScaler(),
                normalise_method='speaker')
            print()

            labels = sorted([x[:3] for x in dataset.classes])

            df = pd.DataFrame(
                index=pd.RangeIndex(10),
                columns=pd.MultiIndex.from_product(
                    [METRICS, ['all'], labels, range(reps)],
                    names=['metric', 'gender', 'class', 'rep']))
            for rep in range(reps):
                for fold, (train, test) in enumerate(splitter.split(
                        dataset.x, dataset.y, dataset.speaker_group_indices)):
                    x_train, y_train = dataset.x[train], dataset.y[train]
                    x_test, y_test = dataset.x[test], dataset.y[test]

                    for i, (valid, test) in enumerate(splitter.split(
                            x_test, y_test, dataset.speaker_indices[test])):
                        subfold = 2 * fold + i
                        print("Fold {}".format(subfold))

                        x_valid, y_valid = x_test[valid], y_test[valid]
                        x_test2, y_test2 = x_test[test], y_test[test]

                        classifier = cross_validate(param_grid, classifier_fn,
                                                    score_fn, x_train, y_train,
                                                    x_valid, y_valid)

                        y_pred = classifier.predict(x_test2)

                        record_metrics(
                            df, subfold, rep, y_test2, y_pred, len(labels))
            print_results(df)
            df.to_csv(os.path.join(
                RESULTS_DIR, corpus, '{}.csv'.format(config)))


def test_dense_model():
    splitter = LeaveOneGroupOut()
    reps = 1
    for corpus in ['iemocap', 'msp-improv']:
        print("logmel_IS09_func")
        dataset = UtteranceDataset('{}/output/logmel_IS09_func.arff'.format(
            corpus), normaliser=StandardScaler(), normalise_method='speaker')
        print()

        labels = sorted([x[:3] for x in dataset.classes])

        class_weight = dataset.n_instances / \
            (dataset.n_classes * np.bincount(dataset.y.astype(np.int)))
        class_weight = dict(zip(range(dataset.n_classes), class_weight))

        df = pd.DataFrame(
            index=pd.RangeIndex(10),
            columns=pd.MultiIndex.from_product(
                [METRICS, ['all'], labels, range(reps)],
                names=['metric', 'gender', 'class', 'rep']))
        for rep in range(reps):
            for fold, (train, test) in enumerate(splitter.split(
                    dataset.x, dataset.y, dataset.speaker_group_indices)):
                x_train, y_train = dataset.x[train], dataset.y[train]
                x_test, y_test = dataset.x[test], dataset.y[test]

                for i, (valid, test) in enumerate(splitter.split(
                        x_test, y_test, dataset.speaker_indices[test])):
                    subfold = 2 * fold + i
                    print("Fold {}".format(subfold))

                    keras.backend.clear_session()
                    model = aldeneh_dense_model(
                        dataset.n_features, dataset.n_classes)
                    model.compile(
                        loss=keras.losses.SparseCategoricalCrossentropy(
                            name='loss'),
                        optimizer=optimizer_fn(),
                        metrics=tf_classification_metrics())

                    x_valid, y_valid = x_test[valid], y_test[valid]
                    x_test2, y_test2 = x_test[test], y_test[test]

                    train_data = BatchedSequence(
                        x_train, y_train, batch_size=50)
                    valid_data = BatchedSequence(
                        x_valid, y_valid, batch_size=50)
                    test_data = BatchedSequence(
                        x_test2, y_test2, batch_size=50)

                    model.fit(
                        train_data,
                        epochs=50,
                        class_weight=class_weight,
                        validation_data=valid_data,
                        callbacks=callbacks_fn(),
                        verbose=0)

                    y_pred = np.argmax(model.predict(test_data), axis=1)
                    y_test2 = np.concatenate([x[1] for x in test_data])

                    record_metrics(
                        df, subfold, rep, y_test2, y_pred, len(labels))
        print_results(df)
        df.to_csv(os.path.join(RESULTS_DIR, corpus, 'logmel_func.csv'))


def test_conv_models():
    splitter = LeaveOneGroupOut()
    reps = 1
    for corpus in ['iemocap', 'msp-improv']:
        print("")
        dataset = FrameDataset('{}/output/logmel.arff.bin'.format(corpus),
                               normaliser=StandardScaler(),
                               normalise_method='speaker')
        dataset.pad_arrays(32)
        print()

        labels = sorted([x[:3] for x in dataset.classes])

        class_weight = dataset.n_instances / \
            (dataset.n_classes * np.bincount(dataset.y.astype(np.int)))
        class_weight = dict(zip(range(dataset.n_classes), class_weight))

        for n_filters, kernel_size in [(384, 8), (288, 16), (208, 32), (128, 64), (80, 128)]:
            df = pd.DataFrame(
                index=pd.RangeIndex(10),
                columns=pd.MultiIndex.from_product(
                    [METRICS, ['all'], labels, range(reps)],
                    names=['metric', 'gender', 'class', 'rep']))
            for rep in range(reps):
                for fold, (train, test) in enumerate(splitter.split(
                        dataset.x, dataset.y, dataset.speaker_group_indices)):
                    x_train = [dataset.x[i] for i in train]
                    y_train = dataset.y[train]
                    x_test = [dataset.x[i] for i in test]
                    y_test = dataset.y[test]

                    for i, (valid, test) in enumerate(splitter.split(
                            x_test, y_test, dataset.speaker_indices[test])):
                        subfold = 2 * fold + i
                        print("Fold {}".format(subfold))

                        keras.backend.clear_session()
                        model = aldeneh_conv_model(dataset.n_features,
                                                   dataset.n_classes,
                                                   n_filters=n_filters,
                                                   kernel_size=kernel_size)
                        model.compile(
                            loss=keras.losses.SparseCategoricalCrossentropy(
                                name='loss'),
                            optimizer=optimizer_fn(),
                            metrics=tf_classification_metrics())

                        x_valid = [x_test[i] for i in valid]
                        y_valid = y_test[valid]
                        x_test2 = [x_test[i] for i in test]
                        y_test2 = y_test[test]

                        train_data = BatchedFrameSequence(x_train, y_train)
                        valid_data = BatchedFrameSequence(x_valid, y_valid)
                        test_data = BatchedFrameSequence(x_test2, y_test2)

                        model.fit(
                            train_data,
                            epochs=50,
                            class_weight=class_weight,
                            validation_data=valid_data,
                            callbacks=callbacks_fn(),
                            verbose=0)

                        y_pred = np.argmax(model.predict(test_data), axis=1)
                        y_test2 = np.concatenate([x[1] for x in test_data])

                        record_metrics(
                            df, subfold, rep, y_test2, y_pred, len(labels))
            print_results(df)
            df.to_csv(os.path.join(RESULTS_DIR, corpus,
                                   'logmel_{}.csv'.format(kernel_size)))


def test_full_model():
    splitter = LeaveOneGroupOut()
    reps = 1
    for corpus in ['iemocap', 'msp-improv']:
        dataset = FrameDataset('{}/output/logmel.arff.bin'.format(corpus),
                               normaliser=StandardScaler(),
                               normalise_method='speaker')
        dataset.pad_arrays(32)
        print()

        labels = sorted([x[:3] for x in dataset.classes])

        class_weight = dataset.n_instances / \
            (dataset.n_classes * np.bincount(dataset.y.astype(np.int)))
        class_weight = dict(zip(range(dataset.n_classes), class_weight))

        df = pd.DataFrame(
            index=pd.RangeIndex(10),
            columns=pd.MultiIndex.from_product(
                [METRICS, ['all'], labels, range(reps)],
                names=['metric', 'gender', 'class', 'rep']))
        for rep in range(reps):
            for fold, (train, test) in enumerate(splitter.split(
                    dataset.x, dataset.y, dataset.speaker_group_indices)):
                x_train = [dataset.x[i] for i in train]
                y_train = dataset.y[train]
                x_test = [dataset.x[i] for i in test]
                y_test = dataset.y[test]

                for i, (valid, test) in enumerate(splitter.split(
                        x_test, y_test, dataset.speaker_indices[test])):
                    subfold = 2 * fold + i
                    print("Fold {}".format(subfold))

                    keras.backend.clear_session()
                    model = aldeneh_full_model(
                        dataset.n_features, dataset.n_classes)
                    model.compile(
                        loss=keras.losses.SparseCategoricalCrossentropy(
                            name='loss'),
                        optimizer=optimizer_fn(),
                        metrics=tf_classification_metrics())

                    x_valid = [x_test[i] for i in valid]
                    y_valid = y_test[valid]
                    x_test2 = [x_test[i] for i in test]
                    y_test2 = y_test[test]

                    train_data = BatchedFrameSequence(x_train, y_train)
                    valid_data = BatchedFrameSequence(x_valid, y_valid)
                    test_data = BatchedFrameSequence(x_test2, y_test2)

                    model.fit(
                        train_data,
                        epochs=50,
                        class_weight=class_weight,
                        validation_data=valid_data,
                        callbacks=callbacks_fn(),
                        verbose=0)

                    y_pred = np.argmax(model.predict(test_data), axis=1)
                    y_test2 = np.concatenate([x[1] for x in test_data])

                    record_metrics(
                        df, subfold, rep, y_test2, y_pred, len(labels))
        print_results(df)
        df.to_csv(os.path.join(RESULTS_DIR, corpus, 'logmel_full.csv'))


def main():
    os.makedirs(os.path.join(RESULTS_DIR, 'iemocap'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'msp-improv'), exist_ok=True)

    tf.get_logger().setLevel(40)  # ERROR level

    test_svm_models()
    test_dense_model()
    test_conv_models()
    test_full_model()


if __name__ == "__main__":
    main()
