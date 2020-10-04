"""Implementation of the model from [1].

[1] Z. Zhang, B. Wu, and B. Schuller, 'Attention-augmented End-to-end
Multi-task Learning for Emotion Prediction from Speech', in ICASSP 2019
- 2019 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), May 2019, pp. 6705–6709, doi:
10.1109/ICASSP.2019.8682896.
"""

import tensorflow as tf
from tensorflow.keras.layers import (GRUCell, Conv1D, Dense, Dropout, Input, Layer, RNN,
                                     MaxPool1D, Reshape, TimeDistributed)
from tensorflow.keras.models import Model, Sequential

__all__ = ['zhang2019_model']


class Attention1D(Layer):
    def build(self, input_shape: tuple):
        _, _, size = input_shape
        self.weight = self.add_weight('weight', (size, 1))

    def call(self, inputs, **kwargs):
        alpha = tf.matmul(inputs, self.weight)  # (batch, steps, 1)
        alpha = tf.nn.softmax(alpha, axis=-2)
        r = tf.matmul(alpha, inputs, transpose_a=True)  # (batch, 1, size)
        r = tf.squeeze(r, axis=-2)  # (batch, size)
        return r


def zhang2019_model(n_classes: int):
    # Input dimensionality is 640 'features' which are actually 640
    # samples per 40ms segment. Each subsequent vector is shifted 10ms
    # from the previous. We assume the sequences are zero-padded to a
    # multiple of 640 samples.
    inputs = Input((None, 1), name='input')
    frames = tf.signal.frame(inputs, 640, 160, axis=1)
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
    gru = RNN([GRUCell(128), GRUCell(128)], return_sequences=True)(features)

    # Attention
    att = Attention1D()(gru)

    x = Dense(n_classes, activation='softmax')(att)
    return Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    model = zhang2019_model(4)
    model.summary()
