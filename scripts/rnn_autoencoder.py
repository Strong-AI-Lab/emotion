import argparse
from pathlib import Path

import netCDF4
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (RNN, AbstractRNNCell, Bidirectional,
                                     Concatenate, Dense, GRUCell, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

from emotion_recognition.dataset import NetCDFDataset


class RNNCallback(Callback):
    def __init__(self, data: tf.data.Dataset, log_dir: str = 'logs',
                 checkpoint=False):
        super().__init__()
        self.data = data
        if self.data.element_spec[0].shape.ndims != 3:
            raise ValueError("Dataset elements must be batched (i.e. of shape "
                             "[batch, step, freq]).")
        self.log_dir = log_dir
        self.checkpoint = checkpoint
        train_log_dir = str(Path(log_dir) / 'train')
        valid_log_dir = str(Path(log_dir) / 'valid')
        self.train_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_writer = tf.summary.create_file_writer(valid_log_dir)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        with self.valid_writer.as_default():
            tf.summary.scalar('rmse', logs['val_rmse'], step=epoch)

            batch, _ = next(iter(self.data))
            reconstruction, representation = self.model(batch, training=False)
            reconstruction = (reconstruction + 1) / 2
            batch = (batch + 1) / 2
            batch = tf.expand_dims(batch, -1)
            reconstruction = tf.expand_dims(reconstruction, -1)
            images = tf.concat([batch, reconstruction], 2)
            tf.summary.image('reconstruction', reconstruction, step=epoch,
                             max_outputs=16)
            tf.summary.image('combined', images, step=epoch, max_outputs=20)
            tf.summary.histogram('representation', representation, step=epoch)

        with self.train_writer.as_default():
            tf.summary.scalar('rmse', logs['rmse'], step=epoch)

        if (epoch + 1) % 10 == 0 and self.checkpoint:
            save_path = str(Path(self.log_dir) / 'model-{:03d}'.format(
                epoch + 1))
            self.model.save_weights(save_path)


def dropout_gru_cell(units: int, keep_prob: float = 0.8) -> AbstractRNNCell:
    return tf.nn.RNNCellDropoutWrapper(GRUCell(units), 0.8, 0.8)


def make_rnn(units: int = 256, bidirectional: bool = False,
             dropout: float = 0.2) -> RNN:
    rnn = RNN(dropout_gru_cell(units, 1 - dropout), return_sequences=True,
              return_state=True)
    return Bidirectional(rnn) if bidirectional else rnn


class TimeRecurrentAutoencoder(Model):
    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            reconstruction, _ = self(x, training=True)
            loss_batch = tf.sqrt(tf.reduce_mean(tf.square(x - reconstruction),
                                                axis=[1, 2]))
            loss = tf.reduce_mean(loss_batch)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        clipped_gvs = [tf.clip_by_value(x, -2, 2) for x in gradients]
        self.optimizer.apply_gradients(zip(clipped_gvs, trainable_vars))

        return {'rmse': loss}

    def test_step(self, data):
        x, _ = data
        reconstruction, _ = self(x, training=False)
        loss_batch = tf.sqrt(tf.reduce_mean(tf.square(x - reconstruction),
                                            axis=[1, 2]))
        loss = tf.reduce_mean(loss_batch)

        return {'rmse': loss}


def create_rnn_model(input_shape: tuple,
                     units: int = 256,
                     layers: int = 2,
                     bidirectional_encoder: bool = False,
                     bidirectional_decoder: bool = False) -> Model:
    inputs = Input(input_shape)

    # Make encoder layers
    enc1 = make_rnn(units, bidirectional_encoder)
    enc_seq1, enc_hidden1 = enc1(inputs)
    enc_states = [enc_hidden1]
    enc_seqN = enc_seq1
    for _ in range(layers - 1):
        encN = make_rnn(units, bidirectional_encoder)
        enc_seqN, enc_hiddenN = encN(enc_seqN)
        enc_states.append(enc_hiddenN)
    # Concatenate output of layers
    encoder = Concatenate()(enc_states)

    # Fully connected needs to have output dimension equal to dimension of
    # decoder state
    dec_dim = units * layers
    if bidirectional_decoder:
        dec_dim *= 2
    representation = Dense(dec_dim, activation='tanh', name='representation')(
        encoder)

    # Initial state of decoder
    decoder_init = tf.split(representation, layers, axis=-1)
    # Decoder input is reversed and shifted input sequence
    dec_inputs = tf.reverse(inputs[:, 1:, :], axis=[1])
    dec_inputs = tf.pad(dec_inputs, [[0, 0], [1, 0], [0, 0]])

    # Make decoder layers and init with output from fully connected layer
    dec1 = make_rnn(units, bidirectional_decoder)
    dec_seq1, dec_hidden1 = dec1(dec_inputs, initial_state=decoder_init[0])
    dec_sequences = [dec_seq1]
    dec_seqN = dec_seq1
    for i in range(1, layers):
        decN = make_rnn(units, bidirectional_decoder)
        dec_seqN, dec_hiddenN = decN(dec_seqN, initial_state=decoder_init[i])
        dec_sequences.append(dec_seqN)
    # Concatenate output of layers
    decoder = Concatenate()(dec_sequences)

    reconstruction = Dense(input_shape[-1], activation='tanh', use_bias=False,
                           name='reconstruction')(decoder)

    model = TimeRecurrentAutoencoder(
        inputs=inputs, outputs=[reconstruction, representation])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=Path, default='logs/rae',
                        help="Directory to store TensorBoard logs.")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of epochs to train for.")
    parser.add_argument('--dataset', type=Path, required=True,
                        help="File containing spectrogram data.")
    parser.add_argument('--valid_fraction', type=float, default=0.1,
                        help="Fraction of data to use as validation data.")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate.")
    args = parser.parse_args()

    args.logs.parent.mkdir(parents=True, exist_ok=True)

    # dataset = NetCDFDataset('jl/output/spectrogram-120.nc')
    dataset = netCDF4.Dataset(str(args.dataset))
    x = tf.constant(dataset.variables['features'])
    data = tf.data.Dataset.from_tensor_slices((x, x)).shuffle(
        1500, reshuffle_each_iteration=False)
    n_valid = int(len(x) * args.valid_fraction)
    valid_data = data.take(n_valid).batch(64)
    train_data = data.skip(n_valid).take(-1).shuffle(1500).batch(64)

    model = create_rnn_model(x[0].shape)
    model.compile(optimizer=Adam(learning_rate=args.learning_rate, clipnorm=2))
    model.summary()
    model.fit(
        train_data, validation_data=valid_data, epochs=args.epochs, callbacks=[
            RNNCallback(valid_data.take(1), log_dir=str(args.logs))
        ])


if __name__ == "__main__":
    main()
