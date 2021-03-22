"""Implementation of model from [1].

[1] A. O. Iskhakova, D. A. Wolf, R. R. Galin, and M. V. Mamchenko, ‘1-D
convolutional neural network based on the inner ear principle to
automatically assess human’s emotional state’, E3S Web Conf., vol. 224,
p. 01023, 2020, doi: 10.1051/e3sconf/202022401023.
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, Input, MaxPool1D, ReLU,
                                     Reshape)

__all__ = ['model']


def model(n_features: int, n_classes: int) -> Model:
    inputs = Input((n_features,))
    x = Reshape((n_features, 1))(inputs)

    # Layer 1
    x = Conv1D(128, 4, activation='relu', padding='same')(x)

    # Layer 2
    x = Conv1D(128, 4, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.25)(x)
    x = MaxPool1D(2)(x)

    # Layers 3-5
    x = Conv1D(64, 4, activation='relu', padding='same')(x)
    x = Conv1D(64, 4, activation='relu', padding='same')(x)
    x = Conv1D(64, 4, activation='relu', padding='same')(x)

    # Layer 6
    x = Conv1D(64, 4, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.25)(x)
    x = MaxPool1D(2)(x)

    # Layers 7-8
    x = Conv1D(32, 2, activation='relu', padding='same')(x)
    x = Conv1D(32, 4, activation='relu', padding='same')(x)

    # Layer 9
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)

    # Output
    x = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)
