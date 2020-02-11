"""Implementation of the paper

G. Trigeorgis et al., 'Adieu features? End-to-end speech emotion recognition
using a deep convolutional recurrent network', in 2016 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), Shanghai,
2016, pp. 5200–5204, doi: 10.1109/ICASSP.2016.7472669.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from python.tensorflow import BalancedSparseCategoricalAccuracy


class CCCLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        mu_x = tf.reduce_mean(y_true, -1, name='mu_x')
        var_x = tfp.stats.variance(y_true, sample_axis=-1, name='var_x')
        mu_y = tf.reduce_mean(y_pred, -1, name='mu_y')
        var_y = tfp.stats.variance(y_pred, sample_axis=-1, name='var_y')

        phi = var_x + var_y + (mu_x - mu_y)**2
        phi_inv = tf.math.reciprocal_no_nan(phi)
        cov = tfp.stats.covariance(y_true, y_pred, sample_axis=-1, event_axis=None)
        return 1.0 - (2.0 * cov * phi_inv)


def ccc(x, y):
    mu_x = x.mean()
    var_x = x.var()
    mu_y = y.mean()
    var_y = y.var()

    phi = var_x + var_y + (mu_x - mu_y)**2
    cov = np.cov(x, y)[0, 1]
    return 1 - (2 * cov / phi)


class HalfWaveRectifier(keras.layers.Layer):
    def __init__(self, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.where(tf.less_equal(inputs, 0.0), 0.0, inputs, name='rectifier')


class EvenSplit(keras.layers.Layer):
    def __init__(self, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        return tf.split(inputs, 150)


def get_model():
    inputs = keras.layers.Input((96000, 1), name='input')

    noise = keras.layers.GaussianNoise(0.1)(inputs)
    conv1 = keras.layers.Conv1D(40, 80, padding='same', activation='linear', use_bias=False, name='conv_1')(noise)
    dropout1 = keras.layers.Dropout(0.5)(conv1)
    rectifier = HalfWaveRectifier(name='halfwave_rectifier')(dropout1)
    maxpool_time = keras.layers.MaxPool1D(2, name='maxpool_time')(rectifier)

    conv2 = keras.layers.Conv1D(40, 4000, padding='same', activation='linear', use_bias=False, name='conv_2')(maxpool_time)
    dropout2 = keras.layers.Dropout(0.5)(conv2)
    maxpool_channels = keras.layers.MaxPool1D(20, data_format='channels_first', name='maxpool_channels')(dropout2)

    # split = EvenSplit()(maxpool_channels)

    # blstm = keras.layers.Bidirectional(keras.layers.RNN(keras.layers.StackedRNNCells([
    #     keras.layers.LSTMCell(128),
    #     keras.layers.LSTMCell(128)
    # ])), name='blstm')(maxpool_channels)

    # recurrent = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, input_shape=), name='blstm_1')
    # recurrent = keras.layers.Bidirectional(keras.layers.LSTM(128), name='blstm_2')(recurrent)

    split = keras.layers.Reshape((150, 320, 2), name='split150')(maxpool_channels)
    blstm = keras.layers.TimeDistributed(keras.layers.Bidirectional(keras.layers.RNN(keras.layers.StackedRNNCells([
        keras.layers.LSTMCell(128),
        keras.layers.LSTMCell(128)
    ])), name='blstm'))(split)
    x = keras.layers.Dense(1, activation='tanh')(blstm)
    x = keras.layers.Flatten()(x)
    return keras.Model(inputs=inputs, outputs=x)


def main():
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    arr = []
    annot = []
    df = pd.read_csv('semaine/combined/1/emotions.txt', sep=' ', index_col=0)
    df.index = pd.TimedeltaIndex(df.index, unit='s')
    df = df.resample('40ms').mean().ffill()
    with open('semaine/combined/1/user_audio.wav', 'rb') as fid:
        audio, sr = tf.audio.decode_wav(fid.read(), desired_channels=1)
    for i in range(160):
        arr.append(tf.constant(audio[i * 96000:(i + 1) * 96000]))
        time_range = slice(pd.Timedelta(i * 6, 's'), pd.Timedelta((i + 1) * 6, 's'))
        annot.append(df['Activation'][time_range].astype(np.float32))

    model = get_model()
    model.compile(
        loss=CCCLoss(),
        metrics=[
            keras.metrics.MeanSquaredError(),
        ],
        optimizer=keras.optimizers.Adam()
    )
    model.summary()

    audio_data = tf.data.Dataset.from_tensor_slices((arr, annot)).shuffle(5000).batch(4)
    model.fit(audio_data)


if __name__ == "__main__":
    main()
