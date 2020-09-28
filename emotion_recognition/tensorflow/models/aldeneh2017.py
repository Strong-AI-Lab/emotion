from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv1D, Dense, GlobalMaxPool1D, Input,
                                     concatenate)


def full_model(n_features: int, n_classes: int) -> Model:
    """Creates the final model from the Aldeneh et. al. (2017) paper."""
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
    return Model(inputs=inputs, outputs=x, name='aldeneh_full_model')
