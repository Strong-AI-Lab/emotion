"""Custom Keras layers for use in TensorFlow models."""

import tensorflow as tf
from keras.layers import Layer

__all__ = ["Attention1D"]


class Attention1D(Layer):
    """Layer that implements simple weighted pooling using softmax
    attention over a sequence of input vectors.
    """

    def build(self, input_shape: tuple):
        _, _, size = input_shape
        self.weight = self.add_weight("weight", (size, 1))

    def call(self, inputs, **kwargs):
        alpha = tf.matmul(inputs, self.weight)  # (batch, steps, 1)
        alpha = tf.nn.softmax(alpha, axis=-2)
        r = tf.reduce_sum(alpha * inputs, axis=-2)  # (batch, size)
        return r
