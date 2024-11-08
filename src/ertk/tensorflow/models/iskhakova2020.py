"""Implementation of model from [1]_.

References
----------
.. [1] A. O. Iskhakova, D. A. Wolf, R. R. Galin, and M. V. Mamchenko,
       '1-D convolutional neural network based on the inner ear
       principle to automatically assess human's emotional state', E3S
       Web Conf., vol. 224, p. 01023, 2020, doi:
       10.1051/e3sconf/202022401023.
"""

import keras

__all__ = ["model"]


def model(n_features: int, n_classes: int) -> keras.Model:
    inputs = keras.Input((n_features,))
    x = keras.layers.Reshape((n_features, 1))(inputs)

    # Layer 1
    x = keras.layers.Conv1D(128, 4, activation="relu", padding="same")(x)

    # Layer 2
    x = keras.layers.Conv1D(128, 4, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.MaxPool1D(2)(x)

    # Layers 3-5
    x = keras.layers.Conv1D(64, 4, activation="relu", padding="same")(x)
    x = keras.layers.Conv1D(64, 4, activation="relu", padding="same")(x)
    x = keras.layers.Conv1D(64, 4, activation="relu", padding="same")(x)

    # Layer 6
    x = keras.layers.Conv1D(64, 4, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.MaxPool1D(2)(x)

    # Layers 7-8
    x = keras.layers.Conv1D(32, 2, activation="relu", padding="same")(x)
    x = keras.layers.Conv1D(32, 4, activation="relu", padding="same")(x)

    # Layer 9
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)

    # Output
    x = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=x)
