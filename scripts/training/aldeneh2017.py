"""Implementation of the paper,

Z. Aldeneh and E. Mower Provost, 'Using regional saliency for speech emotion
recognition', in IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), 2017, pp. 2741–2745,
doi: 10.1109/ICASSP.2017.7952655
"""

from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from emotion_recognition.classification import (PrecomputedSVC,
                                                SKLearnClassifier,
                                                TFClassifier, print_results,
                                                within_corpus_cross_validation)
from emotion_recognition.dataset import NetCDFDataset
from emotion_recognition.tensorflow.classification import BatchedSequence
from emotion_recognition.tensorflow.models.aldeneh2017 import full_model
from sklearn.metrics import recall_score
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Conv1D, Dense, GlobalMaxPool1D, Input,
                                     concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

RESULTS_DIR = 'results/aldeneh2017'


def get_dense_model(n_features, n_classes):
    inputs = Input(shape=(n_features,), name='input')
    x = Dense(1024, activation='relu', kernel_initializer='he_normal',
              name='dense_1')(inputs)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal',
              name='dense_2')(x)
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal',
              name='emotion_prediction')(x)
    return Model(inputs=inputs, outputs=x, name='aldeneh_dense_model')


def get_conv_model(n_features, n_classes, n_filters=128, kernel_size=8):
    inputs = Input(shape=(None, n_features), name='input')
    x = Conv1D(
        n_filters, kernel_size, activation='relu',
        kernel_initializer='he_normal', name='conv'
    )(inputs)
    x = GlobalMaxPool1D(name='maxpool')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal',
              name='dense_1')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal',
              name='dense_2')(x)
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal',
              name='emotion_prediction')(x)
    return Model(inputs=inputs, outputs=x, name='aldeneh_conv_model')


def get_full_model(n_features, n_classes):
    inputs = Input(shape=(None, n_features), name='input')
    x = Conv1D(384, 8, activation='relu', kernel_initializer='he_normal',
               name='conv8')(inputs)
    c1 = GlobalMaxPool1D(name='maxpool_1')(x)

    x = Conv1D(384, 16, activation='relu', kernel_initializer='he_normal',
               name='conv16')(inputs)
    c2 = GlobalMaxPool1D(name='maxpool_2')(x)

    x = Conv1D(384, 32, activation='relu', kernel_initializer='he_normal',
               name='conv32')(inputs)
    c3 = GlobalMaxPool1D(name='maxpool_3')(x)

    x = Conv1D(384, 64, activation='relu', kernel_initializer='he_normal',
               name='conv64')(inputs)
    c4 = GlobalMaxPool1D(name='maxpool_4')(x)

    x = concatenate(name='concatenate')([c1, c2, c3, c4])
    x = Dense(1024, activation='relu', kernel_initializer='he_normal',
              name='dense_1')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal',
              name='dense_2')(x)
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal',
              name='emotion_prediction')(x)
    return Model(inputs=inputs, outputs=x, name='aldeneh_full_model')


def optimizer_fn():
    return RMSprop(learning_rate=0.0001)


def callbacks_fn():
    return [
        EarlyStopping(monitor='val_uar', patience=10,
                      restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_uar', factor=1 / 1.4, patience=0,
                          mode='max')
    ]


def get_tf_dataset(x: np.ndarray, y: np.ndarray, shuffle=True, batch_size=50):
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


def test_svm_models(dataset: NetCDFDataset, config: str):
    param_grid = ParameterGrid({
        'C': 2.0**np.arange(0, 13, 2),
        'gamma': 2.0**np.arange(-15, -2, 2)
    })

    df = within_corpus_cross_validation(
        SKLearnClassifier(partial(PrecomputedSVC, kernel='rbf',
                                  class_weight='balanced')),
        dataset,
        reps=1,
        splitter=LeaveOneGroupOut(),
        param_grid=param_grid,
        cv_score_fn=partial(recall_score, average='macro')
    )

    print_results(df)
    output_dir = Path(RESULTS_DIR) / dataset.corpus / 'svm'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / '{}.csv'.format(config))


def test_dense_model(dataset: NetCDFDataset, config: str = 'logmel_func'):
    model = get_dense_model(480, 4)
    model.summary()
    del model
    tf.keras.backend.clear_session()

    class_weight = (dataset.n_instances
                    / (dataset.n_classes * dataset.class_counts))
    class_weight = dict(zip(range(dataset.n_classes), class_weight))

    df = within_corpus_cross_validation(
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
    output_dir = Path(RESULTS_DIR) / dataset.corpus / 'dense'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / '{}.csv'.format(config))


def test_conv_models(dataset: NetCDFDataset, config: str = 'logmel'):
    model = get_conv_model(40, dataset.n_classes)
    model.summary()
    del model
    tf.keras.backend.clear_session()

    class_weight = (dataset.n_instances
                    / (dataset.n_classes * dataset.class_counts))
    class_weight = dict(zip(range(dataset.n_classes), class_weight))

    for n_filters, kernel_size in [(384, 8), (288, 16), (208, 32), (128, 64),
                                   (80, 128)]:
        print("(n_filters, kernel_size) = ({}, {})".format(n_filters,
                                                           kernel_size))
        df = within_corpus_cross_validation(
            TFClassifier(partial(
                get_conv_model, dataset.n_features, dataset.n_classes,
                n_filters=n_filters, kernel_size=kernel_size
            )),
            dataset,
            reps=1,
            splitter=LeaveOneGroupOut(),
            n_epochs=50,
            class_weight=class_weight,
            data_fn=get_tf_dataset,
            callbacks=callbacks_fn(),
            optimizer=optimizer_fn(),
            verbose=False
        )

        print_results(df)
        output_dir = Path(RESULTS_DIR) / dataset.corpus / 'conv'
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / '{}_{}.csv'.format(config, kernel_size))


def test_full_model(dataset: NetCDFDataset, config: str = 'logmel'):
    model = full_model(40, dataset.n_classes)
    model.summary()
    del model
    tf.keras.backend.clear_session()

    class_weight = (dataset.n_instances
                    / (dataset.n_classes * dataset.class_counts))
    class_weight = dict(zip(range(dataset.n_classes), class_weight))

    df = within_corpus_cross_validation(
        TFClassifier(partial(full_model, dataset.n_features,
                             dataset.n_classes)),
        dataset,
        reps=1,
        splitter=LeaveOneGroupOut(),
        n_epochs=50,
        class_weight=class_weight,
        data_fn=get_tf_dataset,
        callbacks=callbacks_fn(),
        optimizer=optimizer_fn(),
        verbose=False
    )

    print_results(df)
    output_dir = Path(RESULTS_DIR) / dataset.corpus / 'conv_dense'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / '{}_full.csv'.format(config))


def main():
    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    CORPORA = ['iemocap', 'msp-improv']
    for corpus in CORPORA:
        for config in ['IS09', 'IS13_IS09_func', 'GeMAPS', 'eGeMAPS']:
            print(corpus, config)
            dataset = NetCDFDataset('output/{}/{}.nc'.format(corpus, config))
            dataset.normalise(StandardScaler(), scheme='speaker')
            print()
            test_svm_models(dataset, config)

    for corpus in CORPORA:
        print(corpus, "logmel_IS09_func")
        dataset = NetCDFDataset('output/{}/logmel_IS09_func.nc'.format(corpus))
        dataset.normalise(StandardScaler(), scheme='speaker')
        print()
        test_dense_model(dataset, 'logmel_IS09_func')

    for corpus in CORPORA:
        print(corpus, "logmel")
        dataset = NetCDFDataset('output/{}/logmel.nc'.format(corpus))
        dataset.normalise(StandardScaler(), scheme='speaker')
        dataset.pad_arrays(32)
        print()
        try:
            test_conv_models(dataset, 'logmel')
        except Exception:
            pass

        print(corpus, "logmel_full")
        test_full_model(dataset, 'logmel')
        print()


if __name__ == "__main__":
    main()
