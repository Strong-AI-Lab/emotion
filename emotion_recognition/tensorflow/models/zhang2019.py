"""Implementation of the model from [1].

[1] Z. Zhang, B. Wu, and B. Schuller, 'Attention-augmented End-to-end
Multi-task Learning for Emotion Prediction from Speech', in ICASSP 2019
- 2019 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), May 2019, pp. 6705–6709, doi:
10.1109/ICASSP.2019.8682896.
"""

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (RNN, Conv1D, Dense, Dropout, GRUCell, GRU,
                                     Input, MaxPool1D, Reshape,
                                     TimeDistributed)
from tensorflow.keras.models import Model, Sequential

from .layers import Attention1D
from ..utils import create_tf_dataset

__all__ = ['zhang2019_model', 'create_windowed_dataset']


def create_windowed_dataset(x: np.ndarray,
                            y: np.ndarray,
                            sample_weight: Optional[np.ndarray] = None,
                            batch_size: int = 64,
                            shuffle: bool = True) -> tf.data.Dataset:
    """Creates a non-ragged dataset with zero-padding. This appears to cause
    OOM issues while using tf.signal.frame doesn't.
    """
    arrs = []
    for seq in x:
        seq = np.squeeze(seq)
        arr = np.zeros((500, 640), dtype=np.float32)
        for i in range(0, min(len(seq), 80000), 160):
            idx = i // 160
            maxl = min(len(seq) - i, 640)
            arr[idx, :maxl] = seq[i:i + 640]
        arrs.append(arr)
    arrs = np.array(arrs)
    assert tuple(arrs.shape) == (len(x), 500, 640)
    return create_tf_dataset(arrs, y, sample_weight=sample_weight,
                             batch_size=batch_size, shuffle=shuffle)


def zhang2019_model(n_classes: int):
    # Input dimensionality is 640 'features' which are actually 640
    # samples per 40ms segment. Each subsequent vector is shifted 10ms
    # from the previous. We assume the sequences are zero-padded to a
    # multiple of 640 samples.
    inputs = Input((500, 640), name='input')
    frames = tf.expand_dims(inputs, -1)
    dropout = Dropout(0.1)(frames)

    # Conv + maxpool over the 640 samples
    extraction = Sequential()
    extraction.add(Conv1D(40, 40, padding='same', activation='relu'))
    extraction.add(MaxPool1D(2))
    extraction.add(Conv1D(40, 40, padding='same', activation='relu'))
    extraction.add(MaxPool1D(10, data_format='channels_first'))
    extraction.add(Reshape((1280,)))
    features = TimeDistributed(extraction)(dropout)

    # Sequence modelling
    # gru = GRU(128, return_sequences=True)(features)
    gru = RNN([GRUCell(128), GRUCell(128)], return_sequences=True)(features)

    # Attention
    att = Attention1D()(gru)

    x = Dense(n_classes, activation='softmax')(att)
    return Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    from ..utils import test_fit

    test_fit(zhang2019_model, (500, 640), batch_size=16)
