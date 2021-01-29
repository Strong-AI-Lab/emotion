"""Implementation of the paper [1].

[1] G. Trigeorgis et al., 'Adieu features? End-to-end speech emotion
recognition using a deep convolutional recurrent network', in 2016 IEEE
International Conference on Acoustics, Speech and Signal Processing
(ICASSP), Shanghai, 2016, pp. 5200–5204, doi:
10.1109/ICASSP.2016.7472669.
"""

import numpy as np
import pandas as pd
import soundfile
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (RNN, Bidirectional, Conv1D, Dense,
                                     Dropout, Flatten, GaussianNoise, Input,
                                     LSTMCell, MaxPool1D, Reshape,
                                     TimeDistributed)
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.optimizers import Adam


class CCCLoss(Loss):
    def call(self, y_true, y_pred):
        mu_x = tf.reduce_mean(y_true, -1, name='mu_x')
        var_x = tf.math.reduce_variance(y_true, -1, name='var_x')
        mu_y = tf.reduce_mean(y_pred, -1, name='mu_y')
        var_y = tf.math.reduce_variance(y_pred, -1, name='var_y')

        phi = var_x + var_y + (mu_x - mu_y)**2
        phi_inv = tf.math.reciprocal_no_nan(phi)
        sxy = tf.reduce_mean((y_true - mu_x[:, tf.newaxis])
                             * (y_pred - mu_y[:, tf.newaxis]), -1)
        # Negative so that we can minimise -CCC = maximise CCC
        ccc = -2 * sxy * phi_inv
        return ccc


def get_model():
    inputs = Input((96000, 1), name='input')

    noise = GaussianNoise(0.1)(inputs)
    conv1 = Conv1D(40, 80, padding='same', use_bias=False, name='conv_1')(
        noise)
    dropout1 = Dropout(0.5)(conv1)
    rectifier = tf.maximum(dropout1, 0, name='halfwave_rectifier')
    maxpool_time = MaxPool1D(2, name='maxpool_time')(rectifier)

    conv2 = Conv1D(40, 4000, padding='same', use_bias=False, name='conv_2')(
        maxpool_time)
    dropout2 = Dropout(0.5)(conv2)
    maxpool_channels = MaxPool1D(20, data_format='channels_first',
                                 name='maxpool_channels')(dropout2)

    split = Reshape((150, 320, 2), name='split150')(maxpool_channels)

    # recurrent = Bidirectional(LSTM(128, return_sequences=True),
    #                           name='blstm_1')(maxpool_channels)
    # recurrent = Bidirectional(LSTM(128), name='blstm_2')(recurrent)

    blstm = Bidirectional(RNN([LSTMCell(128), LSTMCell(128)]), name='blstm')
    x = TimeDistributed(blstm)(split)
    x = Dense(1, activation='tanh')(x)
    x = Flatten()(x)
    return Model(inputs=inputs, outputs=x)


def main():
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    df = pd.read_csv('datasets/semaine/combined/1/emotions.csv', header=0,
                     index_col=0)
    df.index = pd.TimedeltaIndex(df.index, unit='s')
    df = df.resample('40ms').mean().ffill()

    audio, _ = soundfile.read('datasets/semaine/combined/1/user.wav',
                              always_2d=True)
    arr = []
    annot = []
    for i in range(100):
        arr.append(tf.constant(audio[i * 96000:(i + 1) * 96000]))
        time_range = slice(pd.Timedelta(i * 6, 's'),
                           pd.Timedelta((i + 1) * 6, 's'))
        annot.append(df['Activation'][time_range].astype(np.float32))
    audio_data = tf.data.Dataset.from_tensor_slices((arr, annot))
    audio_data = audio_data.shuffle(100).batch(4)
    del arr, annot, audio, df

    model = get_model()
    model.compile(loss=CCCLoss(), metrics=[MeanSquaredError()],
                  optimizer=Adam())
    model.summary()

    model.fit(audio_data, epochs=50, verbose=1)


if __name__ == "__main__":
    main()
