"""Basic Transformer model for time series classification."""

import keras
import numpy as np
import tensorflow as tf

__all__ = ["model"]


def _encoder_layer(x, num_heads: int, model_dim: int):
    att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=model_dim)(x, x)
    x = keras.layers.LayerNormalization()(x + att)
    ff = keras.layers.Dense(2048)(x)
    ff = keras.layers.ReLU()(ff)
    ff = keras.layers.Dense(model_dim)(ff)
    return keras.layers.LayerNormalization()(x + ff)


def _positional_encoding(seq_len: int, model_dim: int):
    if model_dim % 2 != 0:
        raise ValueError("model_dim must be divisible by 2.")
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    dims = np.arange(model_dim // 2, dtype=np.float32)[None, :]
    angles = pos / np.power(10000, 2 * dims / model_dim)
    # Need to include batch dimension
    pe = np.empty((1, seq_len, model_dim))
    pe[..., 0::2] = np.sin(angles)
    pe[..., 1::2] = np.cos(angles)
    return tf.constant(pe, dtype=tf.float32)


def model(
    n_features: int,
    n_classes: int,
    enc_layers: int = 2,
    num_heads: int = 4,
    seq_len: int = 512,
    model_dim: int = 512,
    pool: str = "max",
) -> keras.Model:
    inputs = keras.Input((seq_len, n_features))
    x = keras.layers.Dense(model_dim, name="projection")(inputs)
    x = keras.layers.add(
        [x, _positional_encoding(seq_len, model_dim)], name="positional_encoding"
    )
    for _ in range(enc_layers):
        x = _encoder_layer(x, num_heads=num_heads, model_dim=model_dim)
    if pool == "max":
        x = keras.layers.GlobalMaxPooling1D()(x)
    else:
        x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=x)
