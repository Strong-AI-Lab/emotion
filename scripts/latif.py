"""Implementation of the paper

S. Latif, R. Rana, S. Khalifa, R. Jurdak, and J. Epps, 'Direct Modelling of
Speech Emotion from Raw Speech', arXiv:1904.03833 [cs, eess], Jul. 2019.
"""

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
from python.tensorflow import BatchedFrameSequence, tf_classification_metrics

RESULTS_DIR = 'results/latif2019'


def callbacks_fn():
    return [
        keras.callbacks.EarlyStopping(monitor='val_uar', patience=20,
                                      restore_best_weights=True, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_uar', factor=0.5,
                                          patience=5, mode='max')
    ]


def optimizer_fn():
    return keras.optimizers.RMSprop(learning_rate=0.0001)


def loss_fn():
    return keras.losses.SparseCategoricalCrossentropy(name='loss')


def get_model(n_classes):
    inputs = keras.Input((None, 1), name='input')

    conv_1 = keras.layers.Conv1D(40, 400, strides=160, padding='same',
                                 name='conv_1')(inputs)
    conv_1 = keras.layers.BatchNormalization()(conv_1)
    conv_1 = keras.layers.ReLU()(conv_1)
    maxpool1 = keras.layers.MaxPool1D(2, name='maxpool_1')(conv_1)

    conv_2 = keras.layers.Conv1D(40, 240, strides=160, padding='same',
                                 name='conv_2')(inputs)
    conv_2 = keras.layers.BatchNormalization()(conv_2)
    conv_2 = keras.layers.ReLU()(conv_2)
    maxpool2 = keras.layers.MaxPool1D(2, name='maxpool_2')(conv_2)

    conv_3 = keras.layers.Conv1D(40, 1600, strides=160, padding='same',
                                 name='conv_3')(inputs)
    conv_3 = keras.layers.BatchNormalization()(conv_3)
    conv_3 = keras.layers.ReLU()(conv_3)
    maxpool3 = keras.layers.MaxPool1D(2, name='maxpool_3')(conv_3)

    concat = keras.layers.Concatenate()([maxpool1, maxpool2, maxpool3])
    concat = tf.expand_dims(concat, axis=-1)

    conv2d = keras.layers.Conv2D(32, (2, 2), name='conv2d')(concat)
    conv2d = keras.layers.BatchNormalization()(conv2d)
    conv2d = keras.layers.ReLU()(conv2d)

    maxpool2d = keras.layers.MaxPool2D((2, 2), name='maxpool2d')(conv2d)
    flattened = keras.layers.Reshape((-1, 32), name='flatten')(maxpool2d)

    lstm = keras.layers.LSTM(128)(flattened)
    dropout = keras.layers.Dropout(0.3)(lstm)
    dense = keras.layers.Dense(1024, activation='relu')(dropout)
    x = keras.layers.Dense(n_classes, activation='softmax')(dense)
    return keras.Model(inputs=inputs, outputs=x)


def get_tf_dataset(x: list, y: np.ndarray, batch_size=16):
    def ragged_to_dense(x, y):
        return x.to_tensor(), y

    # Sort according to length
    lengths = [len(a) for a in x]
    perm = np.argsort(lengths)

    x = [x[i] for i in perm]
    y = y[perm]

    ragged = tf.RaggedTensor.from_row_lengths(np.concatenate(x),
                                              [len(a) for a in x])
    # Group similar lengths in batches, then shuffle batches
    return tf.data.Dataset.from_tensor_slices(
        (ragged, y)
    ).batch(batch_size).shuffle(500).map(ragged_to_dense)


def test_model():
    splitter = LeaveOneGroupOut()
    reps = 1
    for corpus in ['iemocap', 'msp-improv']:
        print("raw")
        dataset = RawDataset('{}/wav_corpus/'.format(corpus), corpus=corpus,
                             normaliser=StandardScaler(),
                             normalise_method='speaker')
        print()

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
                x_train = [dataset.x[i] for i in train]
                y_train = dataset.y[train]
                x_test = [dataset.x[i] for i in test]
                y_test = dataset.y[test]

                for i, (valid, test) in enumerate(splitter.split(
                        x_test, y_test, dataset.speaker_indices[test])):
                    subfold = 2 * fold + i
                    print("Fold {}".format(subfold))

                    keras.backend.clear_session()
                    model = get_model(dataset.n_classes)
                    model.compile(loss=loss_fn(), optimizer=optimizer_fn(),
                                  metrics=tf_classification_metrics())

                    x_valid = [x_test[i] for i in valid]
                    y_valid = y_test[valid]
                    x_test2 = [x_test[i] for i in test]
                    y_test2 = y_test[test]

                    train_data = get_tf_dataset(x_train, y_train)
                    valid_data = get_tf_dataset(x_valid, y_valid)
                    test_data = get_tf_dataset(x_test2, y_test2)

                    model.fit(
                        train_data,
                        epochs=100,
                        class_weight=class_weight,
                        validation_data=valid_data,
                        callbacks=callbacks_fn(),
                        verbose=1)

                    y_pred = np.argmax(model.predict(test_data), axis=1)
                    y_test2 = np.concatenate([x[1] for x in test_data])

                    record_metrics(
                        df, subfold, rep, y_test2, y_pred, len(labels))
        print_results(df)
        df.to_csv(os.path.join(RESULTS_DIR, corpus, 'logmel_func.csv'))


def main():
    os.makedirs(os.path.join(RESULTS_DIR, 'iemocap'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'msp-improv'), exist_ok=True)

    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    test_model()


if __name__ == "__main__":
    main()
