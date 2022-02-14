import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import History, TensorBoard
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from ertk.tensorflow.utils import (
    DataFunction,
    TFModelFunction,
    init_gpu_memory_growth,
    tf_dataset_gen,
    tf_dataset_mem,
    tf_dataset_mem_ragged,
)
from ertk.train import get_scores
from ertk.utils import ScoreFunction, filter_kwargs


def tf_train_val_test(
    model_fn: TFModelFunction,
    train_data: Tuple[np.ndarray, ...],
    valid_data: Tuple[np.ndarray, ...],
    test_data: Optional[Tuple[np.ndarray, ...]] = None,
    scoring: Union[
        str, List[str], Dict[str, ScoreFunction], Callable[..., float]
    ] = "accuracy",
    data_fn: Union[DataFunction, str] = None,
    batch_size: int = 32,
    verbose: int = 0,
    fit_params: Dict[str, Any] = {},
) -> Dict[str, Union[float, History]]:
    """Trains on given data, using given validation data, and tests on
    given test data.

    Args:
    -----
    model_fn: callable
        The function used to create a compiled Model.
    train_data: tuple
        The training data. Tuple elements will be used as positional
        arguments to data_fn.
    valid_data: tuple
        Validation data.
    test_data: tuple, optional
        Optional test data. If not given, will use validation data as
        test data.
    scoring: str, or callable, or list of str, or dict of str to callable
        The scoring to use. Same requirements as for sklearn
        cross_validate().
    data_fn: str or callable
        A string, or callable that returns a tensorflow.data.Dataset
        instance which yields data batches. The callable should take x,
        y and sample_weight parameters, as well as shuffle and
        batch_size keyword arguments. Valid string values are: {"mem",
        "gen", "mem_ragged"}. "mem" indicates an in-memory Dataset.
        "gen" indicates a generator dataset, and "ragged" indicates
        taking variable-length input.
    fit_params: dict, optional
        Any keyword arguments to supply to the Keras fit() method.
        Default is no keyword arguments.

    Returns:
    --------
    scores: dict
        A dictionary with scorer names as keys and scores as values.
    """
    init_gpu_memory_growth()

    if test_data is None:
        test_data = valid_data

    data_fn = fit_params.pop("data_fn", data_fn)
    batch_size = fit_params.pop("batch_size", batch_size)
    if "sample_weight" in fit_params:
        del fit_params["sample_weight"]  # Part of *_data

    data_fns = {
        "mem": tf_dataset_mem,
        "gen": tf_dataset_gen,
        "mem_ragged": tf_dataset_mem_ragged,
    }
    if data_fn is None:
        if len(train_data[0].shape) == 3:
            data_fn = tf_dataset_gen
        elif len(train_data[0].shape) == 1:
            data_fn = tf_dataset_mem_ragged
        else:
            data_fn = tf_dataset_mem
    elif isinstance(data_fn, str):
        data_fn = data_fns[data_fn]
    elif not callable(data_fn):
        raise ValueError(f"Unsupported value for data_fn {data_fn}")

    logging.debug(f"fit_params={fit_params}")
    logging.debug(f"batch_size={batch_size}")
    logging.debug(f"data_fn={data_fn}")

    tf.keras.backend.clear_session()

    clf = model_fn()
    logging.debug(clf)
    if isinstance(clf, Pipeline):
        new_pipeline = clf[:-1]
        clf = clf._final_estimator
        new_pipeline.fit(train_data[0], y=train_data[1])
        train_data = (new_pipeline.transform(train_data[0]), *train_data[1:])
        valid_data = (new_pipeline.transform(valid_data[0]), *valid_data[1:])
        test_data = (new_pipeline.transform(test_data[0]), *test_data[1:])
    if not isinstance(clf, tf.keras.models.Model):
        raise TypeError(
            "model_fn must return a Model or a Pipeline which ends in a Model."
        )

    train_dataset = data_fn(*train_data, batch_size=batch_size)
    valid_dataset = data_fn(*valid_data, batch_size=batch_size, shuffle=False)
    test_dataset = data_fn(*test_data, batch_size=batch_size, shuffle=False)

    clf.summary(print_fn=logging.info)

    fit_params = filter_kwargs(fit_params, clf.fit)
    history = clf.fit(
        train_dataset, validation_data=valid_dataset, verbose=verbose, **fit_params
    )
    # Setting y_true is necessary for dataset creation routines that may
    # reorder data on creation (but without shuffling each time).
    y_true = np.concatenate([x[1] for x in test_dataset.as_numpy_iterator()])
    y_pred = tf.argmax(clf.predict(test_dataset), axis=-1)

    scores = get_scores(scoring, y_pred, y_true)
    scores = {f"test_{k}": v for k, v in scores.items()}
    scores["history"] = history
    return scores


def tf_cross_validate(
    model_fn: TFModelFunction,
    x: np.ndarray,
    y: np.ndarray,
    *,
    cv: BaseCrossValidator,
    groups: Optional[np.ndarray] = None,
    verbose: int = 0,
    scoring: Union[
        str, List[str], Dict[str, ScoreFunction], Callable[..., float]
    ] = "accuracy",
    fit_params: Dict[str, Any] = {},
):
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
        The CV split generator to use.
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
    init_gpu_memory_growth()

    log_dir = fit_params.pop("log_dir", None)
    sw = fit_params.pop("sample_weight", None)
    data_fn = fit_params.pop("data_fn", None)

    logging.debug(f"log_dir={log_dir}")
    logging.debug(f"sample_weight={sw}")
    logging.debug(f"data_fn={data_fn}")
    logging.debug(f"cv={cv}")
    logging.debug(f"fit_params={fit_params}")

    n_folds = cv.get_n_splits(x, y, groups)
    scores = defaultdict(list)
    for fold, (train, test) in enumerate(cv.split(x, y, groups)):
        if verbose:
            logging.info(f"Fold {fold + 1}/{n_folds}")

        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        sw_train = sw[train] if sw is not None else None
        sw_test = sw[test] if sw is not None else None

        callbacks = []
        if log_dir is not None:
            callbacks.append(
                TensorBoard(
                    log_dir=log_dir / str(fold + 1),
                    profile_batch=0,
                    write_graph=False,
                    write_images=False,
                )
            )

        fit_params["callbacks"] = callbacks
        _scores = tf_train_val_test(
            model_fn,
            train_data=(x_train, y_train, sw_train),
            valid_data=(x_test, y_test, sw_test),
            test_data=(x_test, y_test, sw_test),
            data_fn=data_fn,
            scoring=scoring,
            verbose=verbose,
            fit_params=fit_params,
        )

        for k in _scores:
            scores[k].append(_scores[k])
    return {k: np.array(scores[k]) for k in scores}


class TFClassifierWrapper(ClassifierMixin, BaseEstimator):
    """Class wrapper for a TensorFlow Keras classifier model.

    Parameters:
    -----------
    model_fn: callable
        A callable that returns a compiled Model that can be trained.
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
    verbose: bool, default = False
        Whether to output details per epoch.
    """

    def __init__(
        self,
        model_fn: TFModelFunction,
        data_fn: Optional[DataFunction] = None,
        fit_params: Dict = {},
        val_method: Optional[str] = None,
    ):
        self.model_fn = model_fn
        self._data_fn = data_fn
        self.fit_params = fit_params.copy()
        self.val_method = val_method

    def data_fn(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sw: Optional[np.ndarray] = None,
        *,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        if self._data_fn is not None:
            return self._data_fn(x, y, sw, shuffle=shuffle)
        if sw is not None:
            dataset = tf.data.Dataset.from_tensor_slices((x, y, sw))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            dataset = dataset.shuffle(len(x))
        return dataset

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ):
        # Clear graph
        tf.keras.backend.clear_session()
        self.model = self.model_fn()

        train_data = self.data_fn(x, y, sample_weight, shuffle=True)
        if self.val_method is not None:
            valid_data = train_data
        else:
            valid_data = None
        return self.model.fit(
            train_data,
            validation_data=valid_data,
            **self.fit_params,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        test_data = self.data_fn(x, np.zeros(len(x)), shuffle=False)
        y_pred = np.argmax(self.model.predict(test_data), axis=-1)
        return y_pred


class BalancedSparseCategoricalAccuracy(SparseCategoricalAccuracy):
    """Calculates categorical accuracy with class weights inversely
    proportional to their size. This behaves as if classes are balanced
    having the same number of instances, and is equivalent to the
    arithmetic mean recall over all classes.
    """

    def __init__(self, name="balanced_sparse_categorical_accuracy", **kwargs):
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


def tf_classification_metrics():
    return [
        SparseCategoricalAccuracy(name="war"),
        BalancedSparseCategoricalAccuracy(name="uar"),
    ]
