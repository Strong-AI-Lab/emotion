#!/usr/bin/python

from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from tensorflow import keras

from python.classification import TFClassifier, print_results, test_model
from python.dataset import UtteranceDataset


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
    if shuffle:
        data = data.shuffle(10000, reshuffle_each_iteration=True)
    return data.batch(512).prefetch(2)


def main():
    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    model = get_model(88, 4)
    model.summary()
    del model

    dataset = UtteranceDataset('iemocap/output/eGeMAPSv01a.arff')

    class_weight = ((dataset.n_instances / dataset.n_classes)
                    / np.bincount(dataset.y.astype(np.int)))
    class_weight = dict(zip(range(dataset.n_classes), class_weight))

    df = test_model(
        TFClassifier(partial(get_model, dataset.n_features,
                             dataset.n_classes)),
        dataset, splitter=LeaveOneGroupOut(), data_fn=get_tf_dataset,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_uar', patience=20,
                restore_best_weights=True, mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_uar', factor=0.5, patience=5,
                mode='max'
            )
        ], class_weight=class_weight,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001), verbose=0
    )
    print_results(df)


if __name__ == "__main__":
    main()
