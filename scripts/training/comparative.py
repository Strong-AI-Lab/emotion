import argparse
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Concatenate, Conv1D, Dense, Dropout,
                                     GlobalMaxPool1D, Input)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from emotion_recognition.classification import (PrecomputedSVC,
                                                SKLearnClassifier,
                                                TFClassifier, print_results,
                                                test_model)
from emotion_recognition.dataset import (FrameDataset, LabelledDataset,
                                         NetCDFDataset, UtteranceDataset)

RESULTS_DIR = 'results/comparative2020'


def get_mlp_keras_model(n_features: int, n_classes: int,
                        layers: int = 1) -> Model:
    """Creates a Keras model with hidden layers and sigmoid activation, without
    dropout.
    """
    inputs = Input((n_features,), name='input')
    dense_1 = Dense(512, activation='sigmoid')(inputs)
    x = Dense(n_classes, activation='softmax')(dense_1)
    return Model(inputs=inputs, outputs=x)


def get_dense_keras_model(n_features: int, n_classes: int,
                          layers: int = 1) -> Model:
    """Creates a Keras model with hidden layers and ReLU activation, with
    dropout.
    """
    inputs = Input((n_features,), name='input')
    x = inputs
    for _ in range(layers):
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)


def get_aldeneh_full_model(n_features: int, n_classes: int) -> Model:
    """Creates the final model from the Aldeneh et. al. paper."""
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

    x = Concatenate(name='concatenate')([c1, c2, c3, c4])
    x = Dense(1024, activation='relu', kernel_initializer='he_normal',
              name='dense_1')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal',
              name='dense_2')(x)
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal',
              name='emotion_prediction')(x)
    return Model(inputs=inputs, outputs=x, name='aldeneh_full_model')


def get_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int = 32,
                   shuffle: bool = True) -> tf.data.Dataset:
    """Returns a TensorFlow Dataset instance with the given x and y.
    x is assumed to be a 2-D array of shape (n_instances, n_features), and y a
    1-D array of length n_instances.
    """
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        data = data.shuffle(len(x), reshuffle_each_iteration=True)
    return data.batch(batch_size).prefetch(8)


def get_tf_dataset_ragged(x: np.ndarray, y: np.ndarray, batch_size: int = 50,
                          shuffle: bool = True) -> tf.data.Dataset:
    """Returns a TensorFlow Dataset instance from the ragged x and y.
    x is assumed to be a 3-D array of shape (n_instances, length[i],
    n_features), while y is a 1-D array of length n_instances.
    """
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


def get_svm_classifier(dataset, kind='linear') -> SKLearnClassifier:
    kwargs = {}
    param_grid = {'C': 2.0**np.arange(-6, 7, 2)}
    if kind == 'linear':
        kwargs = {'kernel': 'poly', 'degree': 1, 'coef0': 0}
    elif kind == 'poly2':
        kwargs = {'kernel': 'poly', 'degree': 2}
        param_grid['coef0'] = [-1, 0, 1]
    elif kind == 'poly3':
        kwargs = {'kernel': 'poly', 'degree': 3}
        param_grid['coef0'] = [-1, 0, 1]
    elif kind == 'rbf':
        kwargs = {'kernel': 'rbf'}
        param_grid['gamma'] = 2.0**np.arange(-12, -1, 2)

    model_fn = partial(PrecomputedSVC, **kwargs, class_weight='balanced')
    score_fn = partial(recall_score, average='macro')
    classifier = SKLearnClassifier(model_fn, param_grid=param_grid,
                                   cv_score_fn=score_fn)
    return classifier


def get_dense_classifier(dataset, kind='basic') -> TFClassifier:
    class_weight = ((dataset.n_instances / dataset.n_classes)
                    / np.bincount(dataset.y.astype(np.int)))
    class_weight = dict(zip(range(dataset.n_classes), class_weight))

    if kind == 'basic':
        model_fn = partial(get_mlp_keras_model, dataset.n_features,
                           dataset.n_classes)
    elif kind == '1layer':
        model_fn = partial(get_dense_keras_model, dataset.n_features,
                           dataset.n_classes, layers=1)
    elif kind == '2layer':
        model_fn = partial(get_dense_keras_model, dataset.n_features,
                           dataset.n_classes, layers=2)
    elif kind == '3layer':
        model_fn = partial(get_dense_keras_model, dataset.n_features,
                           dataset.n_classes, layers=3)
    else:
        raise NotImplementedError("Other kinds of dense model are not "
                                  "currently implemented.")

    classifier = TFClassifier(
        model_fn, n_epochs=100, class_weight=class_weight,
        data_fn=get_tf_dataset, callbacks=[
            EarlyStopping(monitor='val_uar', patience=20,
                          restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_uar', factor=0.5, patience=5,
                              mode='max')
        ],
        loss=SparseCategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.0001),
        verbose=False
    )
    return classifier


def get_conv_classifier(dataset, kind='aldeneh') -> TFClassifier:
    class_weight = ((dataset.n_instances / dataset.n_classes)
                    / np.bincount(dataset.y.astype(np.int)))
    class_weight = dict(zip(range(dataset.n_classes), class_weight))

    if kind == 'aldeneh':
        model_fn = partial(get_aldeneh_full_model, dataset.n_features,
                           dataset.n_classes)
    else:
        raise NotImplementedError("Other kinds of convolutional model are not "
                                  "currently implemented.")

    model = model_fn()
    model.summary()
    del model
    tf.keras.backend.clear_session()

    classifier = TFClassifier(
        model_fn,
        n_epochs=100,
        data_fn=get_tf_dataset_ragged,
        callbacks=[
            EarlyStopping(monitor='val_uar', patience=20,
                          restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_uar', factor=0.5, patience=5,
                              mode='max')
        ],
        class_weight=class_weight,
        loss=SparseCategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.0001),
        verbose=False
    )
    return classifier


def test_classifier(clf: str,
                    kind: str,
                    dataset: LabelledDataset,
                    reps: int = 1,
                    resultname: Optional[str] = None):
    splitter = LeaveOneGroupOut()
    if dataset.n_speakers > 12:
        splitter = GroupKFold(6)

    if clf == 'svm':
        classifier = get_svm_classifier(dataset, kind=kind)
    elif clf == 'dnn':
        classifier = get_dense_classifier(dataset, kind=kind)
    elif clf == 'cnn':
        classifier = get_conv_classifier(dataset, kind=kind)
    else:
        raise ValueError("--clf must be one of {svm, dnn, cnn}.")

    df = test_model(
        classifier, dataset, reps=reps, splitter=splitter, mode='all',
        genders=['all'], validation='valid'
    )

    print_results(df)
    if resultname:
        output_dir = Path(RESULTS_DIR) / dataset.corpus / clf / kind
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(output_dir) / '{}.csv'.format(resultname))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True,
                        help="The corpus to test.")
    parser.add_argument('--clf', type=str, required=True,
                        help="The type of classifier: {svm, dnn, cnn}.")
    parser.add_argument('--data', type=Path, required=True,
                        help="The data to use.")
    parser.add_argument('--kind', type=str, required=True,
                        help="The kind of classifier.")
    parser.add_argument(
        '--datatype', type=str, default='utterance', required=True,
        help="The type of data: {frame, utterance, netCDF}."
    )
    parser.add_argument('--name', type=str, help="The results output name.")
    parser.add_argument('--noresults', action='store_true',
                        help="Don't output results to file")
    parser.add_argument('--reps', type=int, default=1,
                        help="The number of repetitions to do per test.")
    args = parser.parse_args()

    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.datatype == 'netCDF':
        dataset = NetCDFDataset(
            args.data, normaliser=StandardScaler(),
            normalise_method='speaker'
        )
    elif args.datatype == 'utterance':
        dataset = UtteranceDataset(args.data, normaliser=StandardScaler(),
                                   normalise_method='speaker')
    elif args.datatype == 'frame':
        dataset = FrameDataset(args.data, normaliser=StandardScaler(),
                               normalise_method='speaker')
        dataset.pad_arrays(64)
    else:
        raise ValueError(
            "--datatype must be one of {netCDF, utterance, frame}.")

    if not args.noresults:
        resultname = args.name or args.data.stem
    else:
        resultname = None
    test_classifier(args.clf, args.kind, dataset, reps=args.reps,
                    resultname=resultname)


if __name__ == "__main__":
    main()
