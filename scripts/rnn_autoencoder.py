import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (RNN, AbstractRNNCell, Bidirectional,
                                     Concatenate, Dense, GRUCell, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class FeedRNNCell(AbstractRNNCell):
    pass


def dropout_gru_cell(units: int) -> AbstractRNNCell:
    return tf.nn.RNNCellDropoutWrapper(GRUCell(units), 0.8, 0.8)


def make_rnn(units: int = 256, bidirectional: bool = False) -> RNN:
    rnn = RNN(dropout_gru_cell(units), return_sequences=True,
              return_state=True)
    return Bidirectional(rnn) if bidirectional else rnn


def get_model(input_shape: tuple,
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
    dense = Dense(dec_dim, activation='tanh')(encoder)

    # Initial state of decoder
    layers_init = tf.split(dense, layers, axis=-1)
    # Decoder input is reversed and shifted input sequence
    dec_inputs = tf.reverse(inputs, axis=-1)
    dec_inputs = tf.pad(dec_inputs, [[1, 0], [0, 0]])

    # Make decoder layers and init with output from fully connected layer
    dec1 = make_rnn(units, bidirectional_decoder)
    dec_seq1, dec_hidden1 = dec1(dec_inputs, initial_state=layers_init[0])
    dec_sequences = [dec_seq1]
    dec_seqN = dec_seq1
    for i in range(layers - 1):
        decN = make_rnn(units, bidirectional_decoder)
        dec_seqN, dec_hiddenN = decN(dec_seqN,
                                     initial_state=layers_init[i + 1])
        dec_sequences.append(dec_seqN)
    # Concatenate output of layers
    decoder = Concatenate()(dec_sequences)

    proj = Dense(input_shape[-1], activation='tanh', use_bias=False)(decoder)

    model = Model(inputs=inputs, outputs=proj)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    return model


def main():
    model = get_model()


if __name__ == "__main__":
    main()
