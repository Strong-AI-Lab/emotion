#!/usr/bin/python


import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from python.classification import (PrecomputedSVC, cross_validate,
                                   print_results, record_metrics, METRICS)
from python.dataset import RawDataset, UtteranceDataset
from python.tensorflow import BatchedSequence, tf_classification_metrics


def callbacks_fn():
    return [
        keras.callbacks.EarlyStopping(monitor='val_uar', patience=20,
                                      restore_best_weights=True, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_uar', factor=0.5,
                                          patience=5, mode='max')
    ]


def optimizer_fn():
    return keras.optimizers.Adam(learning_rate=0.001)


def loss_fn():
    return keras.losses.SparseCategoricalCrossentropy(name='loss')


def get_model(n_features, n_classes):
    inputs = keras.Input((n_features,), name='input')
    dense_1 = keras.layers.Dense(512, activation='relu')(inputs)
    dropout_1 = keras.layers.Dropout(0.5)(dense_1)
    dense_2 = keras.layers.Dense(512, activation='relu')(dropout_1)
    dropout_2 = keras.layers.Dropout(0.5)(dense_2)
    dense_3 = keras.layers.Dense(512, activation='relu')(dropout_2)
    dropout_3 = keras.layers.Dropout(0.5)(dense_3)
    x = keras.layers.Dense(n_classes, activation='softmax')(dropout_1)
    return keras.Model(inputs=inputs, outputs=x)


def get_tf_dataset(x, y, shuffle=True):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.shuffle(10000, reshuffle_each_iteration=shuffle)
    return data.batch(512).prefetch(2)


def test_model():
    splitter = LeaveOneGroupOut()
    reps = 1

    dataset = UtteranceDataset('iemocap/output/eGeMAPSv01a.arff')
    labels = sorted([x[:3] for x in dataset.classes])

    class_weight = ((dataset.n_instances / dataset.n_classes)
                    / np.bincount(dataset.y.astype(np.int)))
    class_weight = dict(zip(range(dataset.n_classes), class_weight))

    df = pd.DataFrame(
        index=pd.RangeIndex(10),
        columns=pd.MultiIndex.from_product(
            [METRICS, ['all'], labels, range(reps)],
            names=['metric', 'gender', 'class', 'rep']))
    for rep in range(reps):
        for fold, (train, test) in enumerate(splitter.split(
                dataset.x, dataset.y, dataset.speaker_group_indices)):
            x_train = dataset.x[train]
            y_train = dataset.y[train]
            x_test = dataset.x[test]
            y_test = dataset.y[test]

            for i, (valid, test) in enumerate(splitter.split(
                    x_test, y_test, dataset.speaker_indices[test])):
                subfold = 2 * fold + i
                print("Fold {}".format(subfold))

                keras.backend.clear_session()
                model = get_model(dataset.n_features, dataset.n_classes)
                model.compile(loss=loss_fn(), optimizer=optimizer_fn(),
                              metrics=tf_classification_metrics())

                x_valid = x_test[valid]
                y_valid = y_test[valid]
                x_test2 = x_test[test]
                y_test2 = y_test[test]

                train_data = get_tf_dataset(x_train, y_train)
                valid_data = get_tf_dataset(x_valid, y_valid)
                test_data = get_tf_dataset(x_test2, y_test2, shuffle=False)

                model.fit(
                    train_data,
                    epochs=100,
                    class_weight=class_weight,
                    validation_data=valid_data,
                    callbacks=callbacks_fn(),
                    verbose=0
                )

                y_pred = np.argmax(model.predict(test_data), axis=1)
                y_test2 = np.concatenate([y for x, y in test_data])

                record_metrics(df, subfold, rep, y_test2, y_pred, len(labels))
    print_results(df)


def main():
    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    model = get_model(88, 4)
    model.summary()
    del model

    test_model()


if __name__ == "__main__":
    main()
