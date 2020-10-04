"""Implementation of the model from [1].

[1] S. Latif, R. Rana, S. Khalifa, R. Jurdak, and J. Epps, 'Direct
Modelling of Speech Emotion from Raw Speech', arXiv:1904.03833 [cs,
eess], Jul. 2019, Accessed: Feb. 10, 2020. [Online]. Available:
http://arxiv.org/abs/1904.03833.
"""

import tensorflow as tf
from tensorflow.keras.layers import (LSTM, BatchNormalization, Conv1D, Conv2D,
                                     Dense, Dropout, Input, MaxPool1D,
                                     MaxPool2D, ReLU, Reshape, concatenate)
from tensorflow.keras.models import Model


__all__ = ['latif2019_model']


def latif2019_model(n_classes: int):
    inputs = Input((None, 1), name='input')

    # 3x 1D convs and maxpool over time
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

    # 2D conv and maxpool over features and time
    conv2d = Conv2D(32, (2, 2), name='conv2d', padding='same')(concat)
    conv2d = BatchNormalization()(conv2d)
    conv2d = ReLU()(conv2d)
    maxpool2d = MaxPool2D((2, 2), name='maxpool2d', padding='same')(conv2d)

    # LSTM
    flattened = Reshape((-1, 1920), name='flatten')(maxpool2d)
    lstm = LSTM(128)(flattened)
    dropout = Dropout(0.3)(lstm)

    dense = Dense(1024, activation='relu')(dropout)
    x = Dense(n_classes, activation='softmax')(dense)
    return Model(inputs=inputs, outputs=x)
