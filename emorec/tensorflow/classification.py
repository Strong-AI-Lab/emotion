from collections import defaultdict
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator, LeaveOneGroupOut
from sklearn.model_selection._validation import _score
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, History, TensorBoard
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.utils import Sequence

from ..classification import Classifier, ScoreFunction
from ..utils import batch_arrays, shuffle_multiple
from .utils import DataFunction, TFModelFunction, create_tf_dataset_ragged


class DummyEstimator:
    """Class that implements a dummy estimator for scoring, to avoid
    repeated invocations of `predict()` etc.
    """
    def __init__(self, y_pred):
        self.y_pred = y_pred

    def predict(self, x, **kwargs):
        return self.y_pred

    def predict_proba(self, x, **kwargs):
        return self.y_pred

    def decision_function(self, x, **kwargs):
        return self.y_pred


def fit(model: Model,
        train_data: tf.data.Dataset,
        valid_data: Optional[tf.data.Dataset] = None,
        epochs: int = 1,
        verbose: bool = False,
        **kwargs):
    """Simple fit functions that trains a model with tf.function's."""
    @tf.function
    def train_step(data, use_sample_weight=False):
        if use_sample_weight:
            x, y_true, sample_weight = data
        else:
            x, y_true = data
            sample_weight = None

        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = model.compiled_loss(y_true, y_pred,
                                       sample_weight=sample_weight)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients,
                                            model.trainable_variables))
        model.compiled_metrics.update_state(y_true, y_pred,
                                            sample_weight=sample_weight)
        return loss

    @tf.function
    def test_step(data, use_sample_weight=False):
        if use_sample_weight:
            x, y_true, sample_weight = data
        else:
            x, y_true = data
            sample_weight = None

        y_pred = model(x, training=False)
        loss = model.compiled_loss(y_true, y_pred, sample_weight=sample_weight)
        model.compiled_metrics.update_state(y_true, y_pred,
                                            sample_weight=sample_weight)
        return loss

    use_sample_weight = len(train_data.element_spec) == 3
    iter_fn = partial(tqdm, leave=False) if verbose else iter
    for epoch in range(epochs):
        train_loss = 0.0
        train_metrics = []
        n_batch = 0
        for batch in iter_fn(train_data):
            loss = train_step(batch, use_sample_weight)
            train_loss += loss
            n_batch += 1
        train_loss /= n_batch
        for metric in model.compiled_metrics.metrics:
            train_metrics.append(metric.result())
            metric.reset_states()

        valid_loss = 0.0
        valid_metrics = []
        n_batch = 0
        if valid_data is not None:
            for batch in iter_fn(valid_data):
                loss = test_step(batch, use_sample_weight)
                valid_loss += loss
                n_batch += 1
            valid_loss /= n_batch
            for metric in model.compiled_metrics.metrics:
                valid_metrics.append(metric.result())
                metric.reset_states()

        if verbose:
            msg = "Epoch {:03d}: train_loss = {:.4f}, valid_loss = {:.4f}"
            for metric in model.compiled_metrics.metrics:
                msg += ", train_{0} = {{:.4f}}, valid_{0} = {{:.4f}}".format(
                    metric.name)
            metric_vals = chain(*zip(train_metrics, valid_metrics))
            print(msg.format(epoch, train_loss, valid_loss, *metric_vals))


def tf_train_val_test(model_fn: TFModelFunction,
                      train_data: tf.data.Dataset,
                      valid_data: tf.data.Dataset,
                      test_data: tf.data.Dataset,
                      scoring: Union[str, List[str],
                                     Dict[str, ScoreFunction],
                                     Callable[..., float]] = 'accuracy',
                      **fit_params) -> Dict[str, Union[float, History]]:
    """Trains on given data, using given validation data, and tests on
    given test data.

    Returns:
    --------
    scores, dict
        A dictionary with scorer names as keys and scores as values.
    """
    scores = {}
    tf.keras.backend.clear_session()
    clf = model_fn()

    history = clf.fit(train_data, validation_data=valid_data, **fit_params)
    scores['history'] = history.history

    y_pred = np.argmax(clf.predict(test_data), axis=-1)
    y_true = np.concatenate([x[1] for x in test_data])
    dummy = DummyEstimator(y_pred)
    if isinstance(scoring, str):
        val = get_scorer(scoring)(dummy, None, y_true)
        scores['test_score'] = val
        scores['test_' + scoring] = val
    elif isinstance(scoring, (list, dict)):
        if isinstance(scoring, list):
            scoring = {x: get_scorer(x) for x in scoring}
        _scores = _score(dummy, None, y_true, scoring)
        for k, v in _scores.items():
            scores['test_' + k] = v
    elif callable(scoring):
        scores['test_score'] = scoring(dummy, None, y_true)
    return scores


def tf_cross_validate(model_fn: TFModelFunction,
                      x: np.ndarray,
                      y: np.ndarray,
                      groups: Optional[np.ndarray] = None,
                      cv: BaseCrossValidator = LeaveOneGroupOut(),
                      scoring: Union[str, List[str],
                                     Dict[str, ScoreFunction]] = 'accuracy',
                      data_fn: DataFunction = create_tf_dataset_ragged,
                      sample_weight=None,
                      log_dir: Optional[Path] = None,
                      fit_params: Dict[str, Any] = {}):
    """Performs cross-validation on a TensorFlow model. This works with
    both sequence models and single vector models.

    Args:
    -----
    model_fn: callable,
        The function used to create a compiled Keras model. This is
        called repeatedly on each iteration of cross-validation.
    x: numpy.ndarray,
        The data array. For sequence input this will be a (ragged) 3-D
        array (array of arrays). Otherwise it will be a contiguous 2-D
        matrix.
    y: numpy.ndarray,
        A 1-D array of shape (n_instances,) containing the data labels.
    groups: np.ndarray, optional
        The groups to use for some cross-validation splitters (e.g.
        LeaveOneGroupOut).
    cv: BaseCrossValidator,
        The cross-validator split generator to use. Default is
        LeaveOneGroupOut.
    scoring: str, or callable, or list of str, or dict of str to callable
        The scoring to use. Same requirements as for sklearn
        cross_validate().
    data_fn: callable
        A callable that returns a tensorflow.data.Dataset instance which
        yields data batches. The call signature of data_fn should be
        data_fn(x, y, shuffle=True, **kwargs).
    fit_params: dict, optional
        Any keyword arguments to supply to the Keras fit() method.
        Default is no keyword arguments.
    """
    scores = defaultdict(list)
    n_folds = cv.get_n_splits(x, y, groups)
    for fold, (train, test) in enumerate(cv.split(x, y, groups)):
        print(f"\tFold {fold + 1}/{n_folds}")

        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        if sample_weight is not None:
            sw_train = sample_weight[train]
            sw_test = sample_weight[test]
            train_data = data_fn(x_train, y_train, sample_weight=sw_train,
                                 shuffle=True)
            test_data = data_fn(x_test, y_test, sample_weight=sw_test,
                                shuffle=False)
        else:
            train_data = data_fn(x_train, y_train, shuffle=True)
            test_data = data_fn(x_test, y_test, shuffle=False)

        # Use validation data just for info
        callbacks = []
        if log_dir is not None:
            tb_log_dir = log_dir / str(fold + 1)
            callbacks.append(
                TensorBoard(log_dir=tb_log_dir, profile_batch=0,
                            write_graph=False, write_images=False)
            )

        fit_params['callbacks'] = callbacks
        _scores = tf_train_val_test(
            model_fn, train_data=train_data, valid_data=test_data,
            test_data=test_data, scoring=scoring, **fit_params
        )

        for k in _scores:
            scores[k].append(_scores[k])
    return {k: np.array(scores[k]) for k in scores}


class TFClassifier(Classifier):
    """Class wrapper for a TensorFlow Keras classifier model.

    Parameters:
    -----------
    model_fn: callable
        A callable that returns a new proper classifier that can be trained.
    n_epochs: int, optional, default = 50
        Maximum number of epochs to train for.
    class_weight: dict, optional
        A dictionary mapping class IDs to weights. Default is to ignore
        class weights.
    data_fn: callable, optional
        Callable that takes x and y as input and returns a
        tensorflow.keras.Sequence object or a tensorflow.data.Dataset
        object.
    callbacks: list, optional
        A list of tensorflow.keras.callbacks.Callback objects to use during
        training. Default is an empty list, so that the default Keras
        callbacks are used.
    loss: keras.losses.Loss
        The loss to use. Default is
        tensorflow.keras.losses.SparseCategoricalCrossentropy.
    optimizer: keras.optimizers.Optimizer
        The optimizer to use. Default is tensorflow.keras.optimizers.Adam.
    verbose: bool, default = False
        Whether to output details per epoch.
    """
    def __init__(self, model_fn: TFModelFunction,
                 n_epochs: int = 50,
                 class_weight: Optional[Dict[int, float]] = None,
                 data_fn: Optional[DataFunction] = None,
                 callbacks: List[Callback] = [],
                 loss: Loss = SparseCategoricalCrossentropy(),
                 optimizer: Optimizer = Adam(),
                 verbose: bool = False):
        self.model_fn = model_fn
        self.n_epochs = n_epochs
        self.class_weight = class_weight
        if data_fn is not None:
            self._data_fn = data_fn
        self.callbacks = callbacks
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose

    def data_fn(self, x: np.ndarray, y: np.ndarray,
                shuffle: bool = True) -> tf.data.Dataset:
        if hasattr(self, '_data_fn') and self._data_fn is not None:
            return self._data_fn(x, y, shuffle)
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            dataset = dataset.shuffle(len(x))
        return dataset

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray, fold=0):
        # Clear graph
        tf.keras.backend.clear_session()
        # Reset optimiser and loss
        optimizer = self.optimizer.from_config(self.optimizer.get_config())
        loss = self.loss.from_config(self.loss.get_config())
        for cb in self.callbacks:
            if isinstance(cb, TensorBoard):
                cb.log_dir = str(Path(cb.log_dir).parent / str(fold))

        self.model = self.model_fn()
        self.model.compile(loss=loss, optimizer=optimizer,
                           metrics=tf_classification_metrics())

        train_data = self.data_fn(x_train, y_train, shuffle=True)
        valid_data = self.data_fn(x_valid, y_valid, shuffle=True)
        self.model.fit(
            train_data,
            epochs=self.n_epochs,
            class_weight=self.class_weight,
            validation_data=valid_data,
            callbacks=self.callbacks,
            verbose=int(self.verbose)
        )

    def predict(self, x_test: np.ndarray,
                y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        test_data = self.data_fn(x_test, y_test, shuffle=False)
        y_true = np.concatenate([x[1] for x in test_data])
        y_pred = np.empty_like(y_true)
        np.argmax(self.model.predict(test_data), axis=1, out=y_pred)
        return y_pred, y_true


class BalancedSparseCategoricalAccuracy(SparseCategoricalAccuracy):
    """Calculates categorical accuracy with class weights inversely
    proportional to their size. This behaves as if classes are balanced
    having the same number of instances, and is equivalent to the
    arithmetic mean recall over all classes.
    """
    def __init__(self, name='balanced_sparse_categorical_accuracy', **kwargs):
        super().__init__(name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)


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
    def __init__(self, x: Union[np.ndarray, List[np.ndarray]],
                 y: np.ndarray, prebatched: bool = False, batch_size: int = 32,
                 shuffle: bool = True):
        self.x = x
        self.y = y
        if not prebatched:
            self.x, self.y = batch_arrays(
                self.x, self.y, batch_size=batch_size, shuffle=shuffle)
        if shuffle:
            self.x, self.y = shuffle_multiple(self.x, self.y,
                                              numpy_indexing=True)

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
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32,
                 shuffle: bool = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        if shuffle:
            self.x, self.y = shuffle_multiple(self.x, self.y,
                                              numpy_indexing=True)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx: int):
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        return self.x[sl], self.y[sl]


def tf_classification_metrics():
    return [SparseCategoricalAccuracy(name='war'),
            BalancedSparseCategoricalAccuracy(name='uar')]
