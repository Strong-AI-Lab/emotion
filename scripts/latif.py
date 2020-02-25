"""Implementation of the paper

S. Latif, R. Rana, S. Khalifa, R. Jurdak, and J. Epps, 'Direct Modelling of
Speech Emotion from Raw Speech', arXiv:1904.03833 [cs, eess], Jul. 2019.
"""

import os
from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from python.classification import TFClassifier, print_results, test_model
from python.dataset import RawDataset

RESULTS_DIR = 'results/latif2019'


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


def get_tf_dataset(x: np.ndarray, y: np.ndarray, shuffle=True, batch_size=16):
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


def main():
    os.makedirs(os.path.join(RESULTS_DIR, 'iemocap'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'msp-improv'), exist_ok=True)

    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    for corpus in ['iemocap', 'msp-improv']:
        print("raw")
        dataset = RawDataset('{}/wav_corpus/'.format(corpus), corpus=corpus,
                             normaliser=StandardScaler(),
                             normalise_method='speaker')
        print()

        class_weight = ((dataset.n_instances / dataset.n_classes)
                        / np.bincount(dataset.y.astype(np.int)))
        class_weight = dict(zip(range(dataset.n_classes), class_weight))

        df = test_model(
            TFClassifier(partial(get_model, dataset.n_classes)),
            dataset,
            splitter=LeaveOneGroupOut(),
            data_fn=get_tf_dataset,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_uar', patience=20, restore_best_weights=True,
                    mode='max'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_uar', factor=0.5, patience=5, mode='max')
            ],
            optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
            verbose=True
        )
        print_results(df)
        df.to_csv(os.path.join(RESULTS_DIR, corpus, 'logmel_func.csv'))


if __name__ == "__main__":
    main()
