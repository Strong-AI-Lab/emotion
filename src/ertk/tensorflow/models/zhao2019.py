"""Implementation of the model from [1]_.

.. [1] Z. Zhao et al., "Exploring Deep Spectrum Representations via
       Attention-Based Recurrent and Convolutional Neural Networks for
       Speech Emotion Recognition," IEEE Access, vol. 7, pp.
       97515-97525, 2019, doi: 10.1109/ACCESS.2019.2928625.
"""

import tensorflow as tf
from keras.layers import (
    LSTM,
    RNN,
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    LSTMCell,
    MaxPool2D,
    ReLU,
    Reshape,
    concatenate,
)
from keras.models import Model
from keras.optimizers.adamw import AdamW

from ertk.tensorflow.classification import tf_classification_metrics

from .layers import Attention1D

__all__ = ["model"]


def model(
    n_features: int, n_classes: int, steps: int = 512, learning_rate: float = 0.001
):
    inputs = Input((steps, n_features))
    x = inputs
    norm = LayerNormalization()
    norm.trainable = False
    x = norm(x)

    # BLSTM
    # blstm_seq = Bidirectional(RNN([LSTMCell(128, dropout=0.2), LSTMCell(128)],
    #                               return_sequences=True))(inputs)
    blstm_seq = Bidirectional(LSTM(128, dropout=0.2, return_sequences=True))(x)
    seq_att = Attention1D()(blstm_seq)

    # FCN
    expanded = tf.expand_dims(x, -1)
    conv1 = Conv2D(64, 8)(expanded)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv1 = MaxPool2D(2)(conv1)
    conv1 = Dropout(0.2)(conv1)

    conv2 = Conv2D(128, 5)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = MaxPool2D(2)(conv2)
    conv2 = Dropout(0.2)(conv2)

    conv3 = Conv2D(128, 3)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = MaxPool2D(2)(conv3)
    conv3 = Dropout(0.2)(conv3)

    pool = Reshape((-1, 128))(conv3)
    pool = Dense(128, activation="tanh")(pool)
    fcn_att = Attention1D()(pool)

    concat = concatenate([seq_att, fcn_att])
    x = Dense(128, activation="relu")(concat)
    x = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(
        AdamW(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=tf_classification_metrics(),
        weighted_metrics=[],
    )
    return model
