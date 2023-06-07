import tensorflow as tf

from . import vggish_slim

# These functions are adapted from
# https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_export_tfhub.py


def vggish_definer(variables, checkpoint_path):
    """Defines VGGish with variables tracked and initialized from a checkpoint."""
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)

    def var_tracker(next_creator, **kwargs):
        """Variable creation hook that assigns initial values from a checkpoint."""
        var_name = kwargs["name"]
        var_value = reader.get_tensor(var_name)
        kwargs.update({"initial_value": var_value})
        var = next_creator(**kwargs)
        variables.append(var)
        return var

    def define_vggish(features):
        with tf.variable_creator_scope(var_tracker):
            return vggish_slim.define_vggish_slim(features, training=False)

    return define_vggish


class VGGish(tf.Module):
    """A TF2 Module wrapper around VGGish."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self._variables = []
        self._vggish_fn = tf.compat.v1.wrap_function(
            vggish_definer(self._variables, checkpoint_path),
            signature=(tf.TensorSpec(shape=[None, 96, 64], dtype=tf.float32),),
        )

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, 96, 64], dtype=tf.float32),)
    )
    def __call__(self, waveform):
        return self._vggish_fn(waveform)
