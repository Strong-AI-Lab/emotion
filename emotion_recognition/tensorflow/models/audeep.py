import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (RNN, AbstractRNNCell, Bidirectional,
                                     Dense, GRUCell, Input, concatenate)

__all__ = ['create_trae']


def _dropout_gru_cell(units: int = 256,
                      dropout: float = 0.8) -> AbstractRNNCell:
    return tf.nn.RNNCellDropoutWrapper(GRUCell(units, name='gru_cell'),
                                       1 - dropout, 1 - dropout)


def _make_rnn(units: int = 256, layers: int = 2, bidirectional: bool = False,
              dropout: float = 0.2, name='rnn') -> RNN:
    cells = [_dropout_gru_cell(units, dropout) for _ in range(layers)]
    rnn = RNN(cells, return_sequences=True, return_state=True, name=name)
    return Bidirectional(rnn, name=name) if bidirectional else rnn


def create_trae(input_shape: tuple,
                units: int = 256,
                layers: int = 2,
                bidirectional_encoder: bool = False,
                bidirectional_decoder: bool = False,
                global_batch_size: int = 64
                ) -> Model:
    inputs = Input(input_shape)

    # Make encoder layers
    encoder = _make_rnn(units, layers, bidirectional_encoder, name='encoder')
    enc_states = encoder(inputs)[1:]
    encoder_output = concatenate(enc_states, name='encoder_output_state')

    # Fully connected needs to have output dimension equal to dimension of
    # decoder state
    dec_dim = units * layers
    if bidirectional_decoder:
        dec_dim *= 2
    representation = Dense(dec_dim, activation='tanh', name='representation')(
        encoder_output)

    # Initial state of decoder
    decoder_init_state = tf.split(
        representation,
        2 * layers if bidirectional_decoder else layers,
        axis=-1
    )
    # Decoder input is reversed and shifted input sequence
    targets = inputs[:, :0:-1, :]
    dec_inputs = tf.pad(targets, [[0, 0], [1, 0], [0, 0]],
                        name='decoder_input_sequence')

    # Make decoder layers and init with output from fully connected layer
    decoder = _make_rnn(units, layers, bidirectional_decoder, name='decoder')
    decoder_output = decoder(dec_inputs, initial_state=decoder_init_state)[0]

    n_features = inputs.shape[-1]
    reconstruction = Dense(n_features, activation='tanh',
                           name='reconstruction')(decoder_output)

    model = Model(inputs=inputs, outputs=[reconstruction, representation])
    return model
