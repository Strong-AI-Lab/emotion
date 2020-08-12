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
import time
from pathlib import Path
from typing import Optional

import netCDF4
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (RNN, AbstractRNNCell, Bidirectional,
                                     Dense, GRUCell, Input, concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

from emotion_recognition.utils import shuffle_multiple


class DropoutRNNCell(AbstractRNNCell):
    def __init__(self, cell: AbstractRNNCell, input_dropout: float = 0.2,
                 output_dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell
        self.input_dropout = input_dropout
        self.output_dropout = output_dropout

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size

    def build(self, input_shape):
        return self.cell.build(input_shape)

    def call(self, inputs, states, training=None):
        if training:
            inputs = tf.nn.dropout(inputs, self.input_dropout,
                                   name='input_dropout')
            outputs, states = self.cell.call(inputs, states, training=training)
            outputs = tf.nn.dropout(outputs, self.output_dropout,
                                    name='output_dropout')
            return outputs, states
        else:
            return self.cell.call(inputs, states, training=training)

    def get_config(self):
        config = {
            'input_dropout': self.input_dropout,
            'output_dropout': self.output_dropout,
            'cell': {
                'class_name': self.cell.__class__.__name__,
                'config': self.cell.get_config()
            },
        }
        base_config = super().get_config()
        return dict(**config, **base_config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        from tensorflow.python.keras.layers.serialization import deserialize
        cell = deserialize(config.pop("cell"), custom_objects=custom_objects)
        return cls(cell, **config)


class _RNNCallback(Callback):
    def __init__(self, data: tf.data.Dataset, log_dir: Optional[str] = 'logs',
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
                      dropout: float = 0.8) -> AbstractRNNCell:
    return DropoutRNNCell(GRUCell(units), dropout, dropout)


def _make_rnn(units: int = 256, layers: int = 2, bidirectional: bool = False,
              dropout: float = 0.2, name='rnn') -> RNN:
    cells = [_dropout_gru_cell(units, dropout) for _ in range(layers)]
    rnn = RNN(cells, return_sequences=True, return_state=True, name=name)
    return Bidirectional(rnn, name=name) if bidirectional else rnn


class TimeRecurrentAutoencoder(Model):
    def __init__(self, *args, global_batch_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_batch_size = global_batch_size

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction, _ = self(data, training=True)
            targets = data[:, ::-1, :]
            rmse_batch = tf.sqrt(tf.reduce_mean(
                tf.square(targets - reconstruction), axis=[1, 2]))
            loss = tf.nn.compute_average_loss(
                rmse_batch, global_batch_size=self.global_batch_size)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        clipped_gvs, _ = tf.clip_by_global_norm(gradients, 2)
        self.optimizer.apply_gradients(zip(clipped_gvs, trainable_vars))

        return loss

    @tf.function
    def test_step(self, data):
        reconstruction, _ = self(data, training=False)
        targets = data[:, ::-1, :]
        rmse_batch = tf.sqrt(tf.reduce_mean(
            tf.square(targets - reconstruction), axis=[1, 2]))
        loss = tf.reduce_mean(rmse_batch)

        return loss


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
    dec = _make_rnn(units, layers, bidirectional_decoder, name='decoder')
    dec_seq, *_ = dec(dec_inputs, initial_state=decoder_init)
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
    parser.add_argument('--continue', action='store_true', dest='cont',
                        help="Continue from latest checkpoint.")
    parser.add_argument('--keep_checkpoints', type=int, default=5,
                        help="Number of checkpoints to keep.")
    parser.add_argument('--save_every', type=int, default=10,
                        help="Save every N epochs.")

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

    if args.logs:
        args.logs.parent.mkdir(parents=True, exist_ok=True)

    strategy = tf.distribute.get_strategy()
    if args.multi_gpu:
        strategy = tf.distribute.MirroredStrategy()

    dataset = netCDF4.Dataset(str(args.dataset))
    x = np.array(dataset.variables['features'])
    x = shuffle_multiple(x)[0]
    n_valid = int(len(x) * args.valid_fraction)
    valid_data = tf.data.Dataset.from_tensor_slices(x[:n_valid]).batch(
        args.batch_size)
    train_data = tf.data.Dataset.from_tensor_slices(x[n_valid:]).shuffle(
        len(x)).batch(args.batch_size)

    with strategy.scope():
        model = create_rnn_model(
            x[0].shape, units=args.units, layers=args.layers,
            bidirectional_encoder=args.bidirectional_encoder,
            bidirectional_decoder=args.bidirectional_decoder,
            global_batch_size=args.batch_size
        )
        # Using epsilon=1e-5 seems to offer better stability.
        optimizer = Adam(learning_rate=args.learning_rate, epsilon=1e-5)
        model.compile(optimizer=optimizer)
        checkpoint = tf.train.Checkpoint(model=model)
    model.summary()

    if args.logs:
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, str(args.logs), max_to_keep=args.keep_checkpoints)

    if args.cont:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print("Restoring from checkpoint {}.".format(
            checkpoint_manager.latest_checkpoint))

    @tf.function
    def dist_train_step(batch):
        per_replica_losses = strategy.run(model.train_step, args=(batch,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM,
                               per_replica_losses, axis=None)

    @tf.function
    def dist_test_step(batch):
        per_replica_losses = strategy.run(model.test_step, args=(batch,))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN,
                               per_replica_losses, axis=None)

    callback = _RNNCallback(valid_data.take(1), args.logs,
                            profile=args.profile)
    callback.model = model

    train_data = strategy.experimental_distribute_dataset(train_data)
    valid_data = strategy.experimental_distribute_dataset(valid_data)

    n_train_batches = len(train_data)
    n_valid_batches = len(valid_data)

    for epoch in range(1, args.epochs + 1):
        callback.on_epoch_begin(epoch)
        epoch_time = time.perf_counter()
        train_loss = 0.0
        n_batch = 0
        batch_time = time.perf_counter()
        for batch in tqdm(
                train_data,
                total=n_train_batches,
                desc="Epoch {:03d}".format(epoch),
                unit='batch'):
            callback.on_train_batch_begin(n_batch)
            loss = dist_train_step(batch)
            train_loss += loss
            n_batch += 1
            callback.on_train_batch_end(n_batch, logs={'rmse': loss})
        train_loss /= n_train_batches
        batch_time = time.perf_counter() - batch_time
        batch_time /= n_train_batches

        valid_loss = 0.0
        for batch in valid_data:
            loss = dist_test_step(batch)
            valid_loss += loss
        valid_loss /= n_valid_batches

        epoch_time = time.perf_counter() - epoch_time
        print(
            "Epoch {:03d}: train loss: {:.4f}, valid loss: {:.4f} in {:.2f}s "
            "({:.2f}s per batch).".format(epoch, train_loss, valid_loss,
                                          epoch_time, batch_time)
        )
        callback.on_epoch_end(epoch, logs={'rmse': train_loss,
                                           'val_rmse': valid_loss})
        if args.logs and epoch % args.save_every == 0:
            checkpoint_manager.save(checkpoint_number=epoch)
            print("Saved checkpoint to {}.".format(
                checkpoint_manager.latest_checkpoint))

    if args.logs:
        save_path = str(args.logs / 'model')
        model.save(save_path, include_optimizer=False)
        print("Saved model to {}.".format(save_path))


if __name__ == "__main__":
    main()
