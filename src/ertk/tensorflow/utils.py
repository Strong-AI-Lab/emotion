from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Layer, Wrapper
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.utils import Sequence

from ertk.utils import batch_arrays, shuffle_multiple

TFModelFunction = Callable[..., Union[Model, Pipeline]]
DataFunction = Callable[..., tf.data.Dataset]


def compile_wrap(
    model_fn: Optional[TFModelFunction] = None,
    opt_cls: Type[Optimizer] = Adam,
    opt_kwargs: Dict[str, Any] = dict(learning_rate=0.0001),
    metrics: List[Union[str, Metric]] = ["sparse_categorical_accuracy"],
    loss: Union[str, Loss] = "sparse_categorical_crossentropy",
    **compile_kwargs,
):
    """Wrapper that takes a model creation function and gives a new
    function which returns a compiled model with the given compile
    parameters.

    Args:
    -----
    model_fn: callable, optional
        A method that returns an uncompiled model.
    opt_cls: type
        The Optimizer class to use.
    opt_kwargs: dict
        Keyword arguments to pass to opt_cls.
    metrics: list
        List of metrics to use.
    loss: Loss
        The loss function to use.
    **compile_kwargs: dict
        Other keyword arguments to pass to the model's compile() method.
    """

    def _wrapper(func: Callable[..., Model]):
        @wraps(func)
        def new_model_fn(*args, **kwargs) -> Model:
            model = func(*args, **kwargs)
            model.compile(
                optimizer=opt_cls(**opt_kwargs),
                metrics=metrics,
                loss=loss,
                **compile_kwargs,
            )
            return model

        return new_model_fn

    if model_fn is not None:
        return _wrapper(model_fn)

    return _wrapper


def test_fit(
    model_fn: TFModelFunction,
    input_size: Tuple[int, ...],
    *args,
    batch_size: int = 64,
    num_instances: int = 7000,
    **kwargs,
):
    """Tests the given model architecture/structure by training it on
    dummy data.

    Args:
    -----
    model_fn: callable
        Function that returns a Keras model. Called as model_fn(*args,
        **kwargs).
    input_size: tuple of int
        Input shape to the model. This is used to generate dummy data of
        the correct shape.
    *args
        Positional arguments to pass to model_fn().
    batch_size: int
        The batch size to use.
    num_instances: int
        The number of instances to generate.
    **kwargs
        Keyword arguments to pass to model_fn().
    """
    for gpu in tf.config.get_visible_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    compiled_fn = compile_wrap(model_fn)
    model = compiled_fn(*args, n_classes=7, **kwargs)
    model.summary()

    valid = num_instances // 10
    rng = np.random.default_rng()
    x = rng.normal(size=(num_instances,) + input_size)
    y = rng.integers(7, size=num_instances)
    train_data = tf.data.Dataset.from_tensor_slices((x[valid:], y[valid:]))
    train_data = train_data.batch(batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices((x[:valid], y[:valid]))
    valid_data = valid_data.batch(batch_size)
    model.fit(train_data, validation_data=valid_data, epochs=2, verbose=1)


def tf_dataset_gen(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
):
    """Returns a TensorFlow generator Dataset instance with the given
    data.

    Args:
    -----
    x: numpy.ndarray
        A 2- or 3-D data matrix of shape (n_instances, n_features) or
        (n_instances, seq_len, n_features).
    y: numpy.ndarray
        A 1-D array of length n_instances containing numeric class
        labels.
    sample_weight: numpy.ndarray, optional
        A 1-D array of length n_instances containing sample weights.
        Added as third item in dataset if present.
    batch_size: int
        The batch size to use.
    shuffle: boolean
        Whether or not to shuffle the dataset. Note that shuffling is
        done *before* batching, unlike in `create_tf_dataset_ragged()`.
    """

    def gen_inst():
        if shuffle:
            perm = np.random.permutation(len(x))
        else:
            perm = np.arange(len(x))

        if sample_weight is None:
            for i in perm:
                yield x[i], y[i]
        else:
            for i in perm:
                yield x[i], y[i], sample_weight[i]

    sig: Tuple[tf.TensorSpec, ...] = (
        tf.TensorSpec(shape=x[0].shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )
    if sample_weight is not None:
        sig += (tf.TensorSpec(shape=(), dtype=tf.float32),)
    data = tf.data.Dataset.from_generator(gen_inst, output_signature=sig)
    return data.batch(batch_size).prefetch(2)


def tf_dataset_mem(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Returns a TensorFlow in-memory Dataset instance with the given
    data.

    Args:
    -----
    x: numpy.ndarray
        A 2- or 3-D data matrix of shape (n_instances, n_features) or
        (n_instances, seq_len, n_features).
    y: numpy.ndarray
        A 1-D array of length n_instances containing numeric class
        labels.
    sample_weight: numpy.ndarray, optional
        A 1-D array of length n_instances containing sample weights.
        Added as third item in dataset if present.
    batch_size: int
        The batch size to use.
    shuffle: boolean
        Whether or not to shuffle the dataset. Note that shuffling is
        done *before* batching, unlike in `create_tf_dataset_ragged()`.
    """
    with tf.device("CPU"):
        if sample_weight is None:
            data = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            data = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))

    if shuffle:
        data = data.shuffle(len(x))
    return data.batch(batch_size).prefetch(2)


def tf_dataset_mem_ragged(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Returns a TensorFlow in-memory Dataset instance from
    variable-length features.

    Args:
    -----
    x: numpy.ndarray
        A 3-D data matrix of shape (n_instances, length[i], n_features)
        with variable length axis 1.
    y: numpy.ndarray
        A 1-D array of length n_instances containing numeric class
        labels.
    sample_weight: numpy.ndarray, optional
        A 1-D array of length n_instances containing sample weights.
        Added as third item in dataset if present.
    batch_size: int
        The batch size to use.
    shuffle: boolean
        Whether or not to shuffle the dataset. Note that shuffling is
        done *after* batching, because sequences are sorted by length,
        then batched in similar lengths.
    """

    def ragged_to_dense(x: tf.RaggedTensor, y):
        return x.to_tensor(), y

    def ragged_to_dense_weighted(x: tf.RaggedTensor, y, sample_weight):
        return x.to_tensor(), y, sample_weight

    # Sort according to length
    perm = np.argsort([len(a) for a in x])
    x = x[perm]
    y = y[perm]
    if sample_weight is not None:
        sample_weight = sample_weight[perm]

    ragged = tf.RaggedTensor.from_row_lengths(
        np.concatenate(list(x)), [len(a) for a in x]
    )
    with tf.device("CPU"):
        if sample_weight is None:
            data = tf.data.Dataset.from_tensor_slices((ragged, y))
        else:
            data = tf.data.Dataset.from_tensor_slices((ragged, y, sample_weight))

    # Group similar lengths in batches, then shuffle batches
    data = data.batch(batch_size)
    if shuffle:
        data = data.shuffle(len(x) // batch_size + 1)

    if sample_weight is None:
        data = data.map(ragged_to_dense)
    else:
        data = data.map(ragged_to_dense_weighted)
    return data.prefetch(2)


class BatchedFrameSequence(Sequence):
    """Creates a sequence of batches of frames to process.

    Parameters:
    -----------
    x: ndarray or list of ndarray
        Sequences of vectors.
    y: ndarray
        Labels corresponding to sequences in x.
    prebatched: bool, default = False
        Whether or not x has already been grouped into batches.
    batch_size: int, default = 32
        Batch size to use. Each generated batch will be at most this size.
    shuffle: bool, default = True
        Whether to shuffle the order of the batches.
    """

    def __init__(
        self,
        x: Union[np.ndarray, List[np.ndarray]],
        y: np.ndarray,
        prebatched: bool = False,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.x = x
        self.y = y
        if not prebatched:
            self.x, self.y = batch_arrays(
                self.x, self.y, batch_size=batch_size, shuffle=shuffle
            )
        if shuffle:
            self.x, self.y = shuffle_multiple(self.x, self.y, numpy_indexing=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class BatchedSequence(Sequence):
    """Creates a sequence of batches to process.

    Parameters:
    -----------
    x: ndarray or list of ndarray
        Instance feature vectors. Each vector is assumed to be for a different
        instance.
    y: ndarray
        Labels corresponding to sequences in x.
    prebatched: bool, default = False
        Whether or not x has already been grouped into batches.
    batch_size: int, default = 32
        Batch size to use. Each generated batch will be at most this size.
    shuffle: bool, default = True
        Whether to shuffle the instances.
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        if shuffle:
            self.x, self.y = shuffle_multiple(self.x, self.y, numpy_indexing=True)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx: int):
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        return self.x[sl], self.y[sl]


def print_linear_model_structure(model: Model):
    """Prints the structure of a "sequential" model by listing the layer
    types and shapes in order.

    Args:
    -----
    model: Model
        The model to describe.
    """

    def print_inner(model: Layer, depth: int = 0):
        indent = "\t" * depth
        if not isinstance(model, Model):
            print(indent, model.name, model.output_shape)
            return

        for layer in model.layers:
            name = layer.name
            if name.startswith("tf_op_layer_"):
                name = name[12:]

            print(indent, name, layer.output_shape)
            if isinstance(layer, Model):
                print_inner(layer, depth + 1)
            elif isinstance(layer, Wrapper):
                print_inner(layer.layer, depth + 1)

    print_inner(model)


def init_gpu_memory_growth():
    """Sets TensorFlow to allocate memory on GPU as needed instead of
    all at once.
    """
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
