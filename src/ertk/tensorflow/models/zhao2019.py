"""Implementation of the model from [1]_.

.. [1] Z. Zhao et al., "Exploring Deep Spectrum Representations via
       Attention-Based Recurrent and Convolutional Neural Networks for
       Speech Emotion Recognition," IEEE Access, vol. 7, pp.
       97515-97525, 2019, doi: 10.1109/ACCESS.2019.2928625.
"""

import tensorflow as tf
import keras

from ertk.tensorflow.classification import tf_classification_metrics

from .layers import Attention1D

__all__ = ["model"]


def model(
    n_features: int, n_classes: int, steps: int = 512, learning_rate: float = 0.001
):
    inputs = keras.Input((steps, n_features))
    x = inputs
    norm = keras.layers.LayerNormalization()
    norm.trainable = False
    x = norm(x)

    # BLSTM
    # blstm_seq = Bidirectional(
    #     keras.layers.RNN(
    #         [keras.layers.LSTMCell(128, dropout=0.2), keras.layers.LSTMCell(128)],
    #         return_sequences=True,
    #     )
    # )(inputs)
    blstm_seq = keras.layers.Bidirectional(
        keras.layers.LSTM(128, dropout=0.2, return_sequences=True)
    )(x)
    seq_att = Attention1D()(blstm_seq)

    # FCN
    expanded = tf.expand_dims(x, -1)
    conv1 = keras.layers.Conv2D(64, 8)(expanded)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv1 = keras.layers.MaxPool2D(2)(conv1)
    conv1 = keras.layers.Dropout(0.2)(conv1)

    conv2 = keras.layers.Conv2D(128, 5)(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2 = keras.layers.MaxPool2D(2)(conv2)
    conv2 = keras.layers.Dropout(0.2)(conv2)

    conv3 = keras.layers.Conv2D(128, 3)(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    conv3 = keras.layers.MaxPool2D(2)(conv3)
    conv3 = keras.layers.Dropout(0.2)(conv3)

    pool = keras.layers.Reshape((-1, 128))(conv3)
    pool = keras.layers.Dense(128, activation="tanh")(pool)
    fcn_att = Attention1D()(pool)

    concat = keras.layers.concatenate([seq_att, fcn_att])
    x = keras.layers.Dense(128, activation="relu")(concat)
    x = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=x)
    model.compile(
        keras.optimizers.AdamW(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=tf_classification_metrics(),
        weighted_metrics=[],
    )
    return model
