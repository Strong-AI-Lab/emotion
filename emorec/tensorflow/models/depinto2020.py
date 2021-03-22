"""Implementation of short model from [1].

[1] M. G. de Pinto, M. Polignano, P. Lops, and G. Semeraro, 'Emotions
Understanding Model from Spoken Language using Deep Neural Networks and
Mel-Frequency Cepstral Coefficients', in 2020 IEEE Conference on
Evolving and Adaptive Intelligent Systems (EAIS), May 2020, pp. 1–5,
doi: 10.1109/EAIS48028.2020.9122698.
"""


from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, Input,
                                     Reshape)

__all__ = ['model']


def model(n_features: int, n_classes: int) -> Model:
    inputs = Input((n_features,))
    x = Reshape((n_features, 1))(inputs)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)
