#!/usr/bin/python

"""A TensorFlow 2 implementation of the auDeep representation learning
framework. Original implementation is available at
https://github.com/auDeep/auDeep.

References:
[1] S. Amiriparian, M. Freitag, N. Cummins, and B. Schuller, 'Sequence
to sequence autoencoders for unsupervised representation learning from
audio', 2017.
[2] M. Freitag, S. Amiriparian, S. Pugachevskiy, N. Cummins, and B.
Schuller, 'auDeep: Unsupervised learning of representations from audio
with deep recurrent neural networks', The Journal of Machine Learning
Research, vol. 18, no. 1, pp. 6340–6344, 2017.
"""

import argparse
from pathlib import Path

import netCDF4
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (RNN, AbstractRNNCell, Bidirectional,
                                     Dense, GRUCell, Input, concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

from emotion_recognition.dataset import NetCDFDataset
from emotion_recognition.utils import shuffle_multiple


class _RNNCallback(Callback):
    def __init__(self, data: tf.data.Dataset, log_dir: str = 'logs',
                 checkpoint=False, profile=False):
        super().__init__()
        self.data = data
        if self.data.element_spec.shape.ndims != 3:
            raise ValueError("Dataset elements must be batched (i.e. of shape "
                             "[batch, step, freq]).")
        self.log_dir = log_dir
        self.checkpoint = checkpoint
        self.profile = profile
        self.start_profile = False
        if profile:
            tf.profiler.experimental.start('')
            tf.profiler.experimental.stop(save=False)
        if log_dir:
            train_log_dir = str(Path(log_dir) / 'train')
            valid_log_dir = str(Path(log_dir) / 'valid')
            self.train_writer = tf.summary.create_file_writer(train_log_dir)
            self.valid_writer = tf.summary.create_file_writer(valid_log_dir)
        else:
            self.train_writer = tf.summary.create_noop_writer()
            self.valid_writer = tf.summary.create_noop_writer()

    def on_train_batch_begin(self, batch, logs=None):
        if batch == 0 and self.start_profile:
            tf.profiler.experimental.start(self.log_dir)

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0 and self.start_profile:
            tf.profiler.experimental.stop()
            self.start_profile = False

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 2 and self.profile:
            self.start_profile = True

    def on_epoch_end(self, epoch, logs=None):
        with self.valid_writer.as_default():
            tf.summary.scalar('rmse', logs['val_rmse'], step=epoch)

            batch = next(iter(self.data))
            reconstruction, representation = self.model(batch, training=False)
            reconstruction = reconstruction[:, ::-1, :]
            images = tf.concat([batch, reconstruction], 2)
            images = tf.expand_dims(images, -1)
            images = (images + 1) / 2
            tf.summary.image('combined', images, step=epoch, max_outputs=20)
            tf.summary.histogram('representation', representation, step=epoch)

        with self.train_writer.as_default():
            tf.summary.scalar('rmse', logs['rmse'], step=epoch)

        if (epoch + 1) % 10 == 0 and self.checkpoint:
            save_path = str(Path(self.log_dir) / 'model-{:03d}'.format(
                epoch + 1))
            self.model.save_weights(save_path)


def _dropout_gru_cell(units: int = 256,
                      keep_prob: float = 0.8) -> AbstractRNNCell:
    return tf.nn.RNNCellDropoutWrapper(GRUCell(units), 0.8, 0.8)


def _make_rnn(units: int = 256, layers: int = 2, bidirectional: bool = False,
              dropout: float = 0.2, name='rnn') -> RNN:
    cells = [_dropout_gru_cell(units, 1 - dropout) for _ in range(layers)]
    rnn = RNN(cells, return_sequences=True, return_state=True, name=name)
    return Bidirectional(rnn, name=name) if bidirectional else rnn


class TimeRecurrentAutoencoder(Model):
    def __init__(self, *args, global_batch_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_batch_size = global_batch_size

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction, _ = self(data, training=True)
            targets = data[:, ::-1, :]
            rmse_batch = tf.sqrt(tf.reduce_mean(
                tf.square(targets - reconstruction), axis=[1, 2]))
            loss = tf.nn.compute_average_loss(
                rmse_batch, global_batch_size=self.global_batch_size)
            train_loss = tf.reduce_mean(rmse_batch)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # clipped_gvs = [tf.clip_by_value(x, -2, 2) for x in gradients]
        clipped_gvs, _ = tf.clip_by_global_norm(gradients, 2)
        self.optimizer.apply_gradients(zip(clipped_gvs, trainable_vars))

        return {'rmse': train_loss}

    def test_step(self, data):
        reconstruction, _ = self(data, training=False)
        targets = data[:, ::-1, :]
        rmse_batch = tf.sqrt(tf.reduce_mean(
            tf.square(targets - reconstruction), axis=[1, 2]))
        loss = tf.reduce_mean(rmse_batch)

        return {'rmse': loss}


def create_rnn_model(input_shape: tuple,
                     units: int = 256,
                     layers: int = 2,
                     bidirectional_encoder: bool = False,
                     bidirectional_decoder: bool = False,
                     global_batch_size: int = 64
                     ) -> TimeRecurrentAutoencoder:
    inputs = Input(input_shape)
    time_steps, features = input_shape

    # Make encoder layers
    enc = _make_rnn(units, layers, bidirectional_encoder, name='encoder')
    _, *enc_states = enc(inputs)
    # Concatenate output of layers
    encoder = concatenate(enc_states, name='encoder_output_state')

    # Fully connected needs to have output dimension equal to dimension of
    # decoder state
    dec_dim = units * layers
    if bidirectional_decoder:
        dec_dim *= 2
    representation = Dense(dec_dim, activation='tanh', name='representation')(
        encoder)

    # Initial state of decoder
    decoder_init = tf.split(
        representation, 2 * layers if bidirectional_decoder else layers,
        axis=-1
    )
    # Decoder input is reversed and shifted input sequence
    targets = inputs[:, ::-1, :]
    dec_inputs = targets[:, :time_steps - 1, :]
    dec_inputs = tf.pad(dec_inputs, [[0, 0], [1, 0], [0, 0]],
                        name='decoder_input_sequence')

    # Make decoder layers and init with output from fully connected layer
    dec1 = _make_rnn(units, layers, bidirectional_decoder, name='decoder')
    dec_seq, *_ = dec1(dec_inputs, initial_state=decoder_init)
    # Concatenate output of layers
    decoder = (dec_seq)

    reconstruction = Dense(features, activation='tanh',
                           name='reconstruction')(decoder)

    model = TimeRecurrentAutoencoder(
        global_batch_size=global_batch_size, inputs=inputs,
        outputs=[reconstruction, representation]
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True,
                        help="File containing spectrogram data.")

    parser.add_argument('--logs', type=Path,
                        help="Directory to store TensorBoard logs.")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of epochs to train for.")
    parser.add_argument('--valid_fraction', type=float, default=0.1,
                        help="Fraction of data to use as validation data.")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Global batch size.")
    parser.add_argument('--multi_gpu', action='store_true',
                        help="Use all GPUs for training.")
    parser.add_argument('--profile', action='store_true',
                        help="Profile a batch.")

    parser.add_argument('--units', type=int, default=256,
                        help="Dimensionality of RNN cells.")
    parser.add_argument('--layers', type=int, default=2,
                        help="Number of stacked RNN layers.")
    parser.add_argument('--bidirectional_encoder', action='store_true',
                        help="Use a bidirectional encoder.")
    parser.add_argument('--bidirectional_decoder', action='store_true',
                        help="Use a bidirectional decoder.")
    args = parser.parse_args()

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    args.logs.parent.mkdir(parents=True, exist_ok=True)

    distribute_strategy = tf.distribute.get_strategy()
    if args.multi_gpu:
        distribute_strategy = tf.distribute.MirroredStrategy()

    # dataset = NetCDFDataset('jl/output/spectrogram-120.nc')
    dataset = netCDF4.Dataset(str(args.dataset))
    x = np.array(dataset.variables['features'])
    x = shuffle_multiple(x)[0]
    n_valid = int(len(x) * args.valid_fraction)
    valid_data = tf.data.Dataset.from_tensor_slices(x[:n_valid]).batch(
        args.batch_size)
    train_data = tf.data.Dataset.from_tensor_slices(x[n_valid:]).shuffle(
        len(x)).batch(args.batch_size)

    with distribute_strategy.scope():
        model = create_rnn_model(
            x[0].shape, units=args.units, layers=args.layers,
            bidirectional_encoder=args.bidirectional_encoder,
            bidirectional_decoder=args.bidirectional_decoder,
            global_batch_size=args.batch_size
        )
        # Using epsilon=1e-5 seems to offer better stability.
        model.compile(optimizer=Adam(learning_rate=args.learning_rate,
                                     epsilon=1e-5))
    model.summary()
    model.fit(
        train_data, validation_data=valid_data, epochs=args.epochs, callbacks=[
            _RNNCallback(valid_data.take(1), log_dir=str(args.logs),
                         profile=args.profile)
        ]
    )


if __name__ == "__main__":
    main()
