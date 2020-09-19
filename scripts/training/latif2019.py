"""Implementation of the paper

S. Latif, R. Rana, S. Khalifa, R. Jurdak, and J. Epps, 'Direct Modelling of
Speech Emotion from Raw Speech', arXiv:1904.03833 [cs, eess], Jul. 2019.
"""

from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, BatchNormalization, Conv1D, Conv2D,
                                     Dense, Dropout, Input, MaxPool1D,
                                     MaxPool2D, ReLU, Reshape, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

from emotion_recognition.classification import (TFClassifier, print_results,
                                                within_corpus_cross_validation)
from emotion_recognition.dataset import RawDataset

RESULTS_DIR = 'results/latif2019'


def get_model(n_classes: int):
    inputs = Input((None, 1), name='input')

    conv_1 = Conv1D(40, 400, strides=160, padding='same',
                    name='conv_25ms')(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = ReLU()(conv_1)
    maxpool1 = MaxPool1D(2, name='maxpool_1')(conv_1)

    conv_2 = Conv1D(40, 240, strides=160, padding='same',
                    name='conv_15ms')(inputs)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = ReLU()(conv_2)
    maxpool2 = MaxPool1D(2, name='maxpool_2')(conv_2)

    conv_3 = Conv1D(40, 1600, strides=160, padding='same',
                    name='conv_100ms')(inputs)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = ReLU()(conv_3)
    maxpool3 = MaxPool1D(2, name='maxpool_3')(conv_3)

    concat = concatenate([maxpool1, maxpool2, maxpool3], axis=-1)
    concat = tf.expand_dims(concat, axis=-1)

    conv2d = Conv2D(32, (2, 2), name='conv2d', padding='same')(concat)
    conv2d = BatchNormalization()(conv2d)
    conv2d = ReLU()(conv2d)

    maxpool2d = MaxPool2D((2, 2), name='maxpool2d', padding='same')(conv2d)
    flattened = Reshape((-1, 1920), name='flatten')(maxpool2d)

    lstm = LSTM(128)(flattened)
    dropout = Dropout(0.3)(lstm)
    dense = Dense(1024, activation='relu')(dropout)
    x = Dense(n_classes, activation='softmax')(dense)
    return Model(inputs=inputs, outputs=x)


def get_tf_dataset(x: np.ndarray, y: np.ndarray, shuffle: bool = True,
                   batch_size: int = 16):
    def ragged_to_dense(x, y):
        return x.to_tensor(), y

    # Sort according to length
    slices = np.array([len(a) for a in x])
    perm = np.argsort(slices)
    x = x[perm]
    y = y[perm]
    slices = slices[perm]

    ragged = tf.RaggedTensor.from_row_lengths(np.concatenate(x), slices)
    data = tf.data.Dataset.from_tensor_slices((ragged, y))
    # Group similar lengths in batches, then shuffle batches
    data = data.batch(batch_size)
    if shuffle:
        data = data.shuffle(1000)
    return data.map(ragged_to_dense)


def main():
    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    model = get_model(4)
    model.summary()
    del model
    tf.keras.backend.clear_session()

    for corpus in ['iemocap', 'msp-improv']:
        dataset = RawDataset('datasets/{}/files.txt'.format(corpus),
                             corpus=corpus)
        dataset.pad_arrays()

        class_weight = ((dataset.n_instances / dataset.n_classes)
                        / np.bincount(dataset.y.astype(np.int)))
        class_weight = dict(zip(range(dataset.n_classes), class_weight))

        data_fn = partial(get_tf_dataset, batch_size=64)
        clf = TFClassifier(
            partial(get_model, dataset.n_classes), n_epochs=50,
            class_weight=class_weight, data_fn=data_fn,
            callbacks=[
                EarlyStopping(
                    monitor='val_uar', patience=20, restore_best_weights=True,
                    mode='max'
                ),
                ReduceLROnPlateau(
                    monitor='val_uar', factor=0.5, patience=5, mode='max')
            ],
            optimizer=RMSprop(learning_rate=0.0001),
            verbose=1
        )
        df = within_corpus_cross_validation(clf, dataset,
                                            splitter=LeaveOneGroupOut())
        print_results(df)
        output_dir = Path(RESULTS_DIR) / corpus
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'raw_audio.csv')


if __name__ == "__main__":
    main()
