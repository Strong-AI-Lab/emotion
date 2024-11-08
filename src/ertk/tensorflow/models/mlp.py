"""Basic multi-layer perceptron (MLP) model."""

from collections.abc import Sequence

import keras

__all__ = ["model"]


def model(
    n_features: int,
    n_classes: int,
    units: Sequence[int] = [512],
    dropout: float = 0.5,
) -> keras.Model:
    """Creates a Keras model with hidden layers and ReLU activation,
    with 50% dropout.
    """
    inputs = keras.Input((n_features,), name="input")
    x = inputs
    for n_units in units:
        x = keras.layers.Dense(n_units, activation="relu")(x)
        x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=x)
