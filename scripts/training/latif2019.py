"""Implementation of the paper

S. Latif, R. Rana, S. Khalifa, R. Jurdak, and J. Epps, 'Direct Modelling of
Speech Emotion from Raw Speech', arXiv:1904.03833 [cs, eess], Jul. 2019.
"""

from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from emotion_recognition.classification import (print_results,
                                                within_corpus_cross_validation)
from emotion_recognition.dataset import LabelledDataset
from emotion_recognition.tensorflow.classification import TFClassifier
from emotion_recognition.tensorflow.models import latif2019_model
from emotion_recognition.tensorflow.utils import create_tf_dataset_ragged
from sklearn.model_selection import LeaveOneGroupOut
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop


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

    model = latif2019_model(4)
    model.summary()
    del model
    tf.keras.backend.clear_session()

    for corpus in ['iemocap', 'msp-improv']:
        # dataset = LabelledDataset('output/{}/raw_audio.nc'.format(corpus))
        dataset = LabelledDataset('datasets/{}/files.txt'.format(corpus))
        dataset.pad_arrays()

        class_weight = (dataset.n_instances
                        / (dataset.n_classes * dataset.class_counts))
        class_weight = dict(zip(range(dataset.n_classes), class_weight))

        data_fn = partial(create_tf_dataset_ragged, batch_size=64)
        clf = TFClassifier(
            partial(latif2019_model, dataset.n_classes), n_epochs=50,
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
        output_dir = Path('results') / 'latif2019' / corpus
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'raw_audio.csv')


if __name__ == "__main__":
    main()
