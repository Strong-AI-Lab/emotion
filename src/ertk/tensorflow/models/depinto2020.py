"""Implementation of short model from [1]_.

References
----------
.. [1] M. G. de Pinto, M. Polignano, P. Lops, and G. Semeraro, 'Emotions
       Understanding Model from Spoken Language using Deep Neural
       Networks and Mel-Frequency Cepstral Coefficients', in 2020 IEEE
       Conference on Evolving and Adaptive Intelligent Systems (EAIS),
       May 2020, pp. 1-5, doi: 10.1109/EAIS48028.2020.9122698.
"""

import keras

__all__ = ["model"]


def model(n_features: int, n_classes: int) -> keras.Model:
    inputs = keras.Input((n_features,))
    x = keras.layers.Reshape((n_features, 1))(inputs)
    # x = keras.layers.Conv1D(128, 40, activation="relu", padding="same")(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.MaxPool1D(8)(x)
    x = keras.layers.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=x)
