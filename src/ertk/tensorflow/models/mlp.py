"""Basic multi-layer perceptron (MLP) model."""

from typing import Sequence

from keras.layers import Dense, Dropout, Input
from keras.models import Model

__all__ = ["model"]


def model(
    n_features: int,
    n_classes: int,
    units: Sequence[int] = [512],
    dropout: float = 0.5,
) -> Model:
    """Creates a Keras model with hidden layers and ReLU activation,
    with 50% dropout.
    """
    inputs = Input((n_features,), name="input")
    x = inputs
    for n_units in units:
        x = Dense(n_units, activation="relu")(x)
        x = Dropout(dropout)(x)
    x = Dense(n_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=x)
