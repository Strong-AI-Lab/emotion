"""Implementation of the model from [1]_.

References
----------
.. [1] S. Latif, R. Rana, S. Khalifa, R. Jurdak, and J. Epps, 'Direct
       Modelling of Speech Emotion from Raw Speech', arXiv:1904.03833
       [cs, eess], Jul. 2019, Accessed: Feb. 10, 2020. [Online].
       Available: http://arxiv.org/abs/1904.03833.
"""

import keras
import tensorflow as tf

__all__ = ["model"]


def model(n_classes: int, n_features: int = 1):
    if n_features != 1:
        raise ValueError("This model can only accept waveform input.")

    inputs = keras.Input((None, 1), name="input")

    # 3x 1D convs and maxpool over time
    conv_1 = keras.layers.Conv1D(
        40,
        400,
        strides=160,
        padding="same",
        kernel_initializer="he_normal",
        name="conv_25ms",
    )(inputs)
    conv_1 = keras.layers.BatchNormalization()(conv_1)
    conv_1 = keras.layers.ReLU()(conv_1)
    maxpool1 = keras.layers.MaxPool1D(2, name="maxpool_1")(conv_1)

    conv_2 = keras.layers.Conv1D(
        40,
        240,
        strides=160,
        padding="same",
        kernel_initializer="he_normal",
        name="conv_15ms",
    )(inputs)
    conv_2 = keras.layers.BatchNormalization()(conv_2)
    conv_2 = keras.layers.ReLU()(conv_2)
    maxpool2 = keras.layers.MaxPool1D(2, name="maxpool_2")(conv_2)

    conv_3 = keras.layers.Conv1D(
        40,
        1600,
        strides=160,
        padding="same",
        kernel_initializer="he_normal",
        name="conv_100ms",
    )(inputs)
    conv_3 = keras.layers.BatchNormalization()(conv_3)
    conv_3 = keras.layers.ReLU()(conv_3)
    maxpool3 = keras.layers.MaxPool1D(2, name="maxpool_3")(conv_3)

    concat = keras.layers.concatenate([maxpool1, maxpool2, maxpool3], axis=-1)
    concat = tf.expand_dims(concat, axis=-1)

    # 2D conv and maxpool over features and time
    conv2d = keras.layers.Conv2D(
        32, (2, 2), name="conv2d", padding="same", kernel_initializer="he_normal"
    )(concat)
    conv2d = keras.layers.BatchNormalization()(conv2d)
    conv2d = keras.layers.ReLU()(conv2d)
    maxpool2d = keras.layers.MaxPool2D((2, 2), name="maxpool2d", padding="same")(conv2d)

    # LSTM
    flattened = keras.layers.Reshape((-1, 1920), name="flatten")(maxpool2d)
    lstm = keras.layers.LSTM(128)(flattened)
    dropout = keras.layers.Dropout(0.3)(lstm)

    dense = keras.layers.Dense(
        1024,
        activation="relu",
        kernel_initializer="he_normal",
    )(dropout)
    x = keras.layers.Dense(
        n_classes,
        activation="softmax",
        kernel_initializer="he_normal",
    )(dense)
    return keras.Model(inputs=inputs, outputs=x)
