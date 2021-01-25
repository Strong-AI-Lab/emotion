"""Implementation of the final full model architecture from [1].

[1] Z. Aldeneh and E. Mower Provost, 'Using regional saliency for speech
emotion recognition', in 2017 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), Mar. 2017, pp.
2741–2745, doi: 10.1109/ICASSP.2017.7952655.
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv1D, Dense, GlobalMaxPool1D, Input,
                                     concatenate)

__all__ = ['aldeneh2017_model']


def aldeneh2017_model(n_features: int, n_classes: int) -> Model:
    inputs = Input(shape=(None, n_features), name='input')
    x = Conv1D(384, 8, activation='relu', kernel_initializer='he_normal',
               name='conv8')(inputs)
    c1 = GlobalMaxPool1D(name='maxpool_1')(x)

    x = Conv1D(384, 16, activation='relu', kernel_initializer='he_normal',
               name='conv16')(inputs)
    c2 = GlobalMaxPool1D(name='maxpool_2')(x)

    x = Conv1D(384, 32, activation='relu', kernel_initializer='he_normal',
               name='conv32')(inputs)
    c3 = GlobalMaxPool1D(name='maxpool_3')(x)

    x = Conv1D(384, 64, activation='relu', kernel_initializer='he_normal',
               name='conv64')(inputs)
    c4 = GlobalMaxPool1D(name='maxpool_4')(x)

    x = concatenate([c1, c2, c3, c4], name='concatenate')
    x = Dense(1024, activation='relu', kernel_initializer='he_normal',
              name='dense_1')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal',
              name='dense_2')(x)
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal',
              name='emotion_prediction')(x)
    return Model(inputs=inputs, outputs=x)
