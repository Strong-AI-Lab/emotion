from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator, LeaveOneGroupOut
from tensorflow.keras.callbacks import History, TensorBoard
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from ..dataset import LabelledDataset
from ..utils import get_scores, ScoreFunction
from .utils import DataFunction, TFModelFunction, create_tf_dataset_ragged


def tf_train_val_test(
    model_fn: TFModelFunction,
    train_data: Tuple[np.ndarray, ...],
    valid_data: Tuple[np.ndarray, ...],
    test_data: Optional[Tuple[np.ndarray, ...]] = None,
    data_fn: DataFunction = create_tf_dataset_ragged,
    scoring: Union[
        str, List[str], Dict[str, ScoreFunction], Callable[..., float]
    ] = "accuracy",
    **fit_params,
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
    data_fn: callable
        A callable that returns a tensorflow.data.Dataset instance which
        yields data batches. The call signature of data_fn should be
        data_fn(x, y, shuffle=True, **kwargs).
    fit_params: dict, optional
        Any keyword arguments to supply to the Keras fit() method.
        Default is no keyword arguments.

    Returns:
    --------
    scores: dict
        A dictionary with scorer names as keys and scores as values.
    """
    if test_data is None:
        test_data = valid_data

    clf = TFClassifierWrapper(model_fn, data_fn=data_fn)
    history = clf.fit(train_data, valid_data, **fit_params)
    y_pred, y_true = clf.predict(test_data[0], test_data[1])

    scores = get_scores(scoring, y_pred, y_true)
    scores["history"] = history.history
    return scores


def tf_cross_validate(
    model_fn: TFModelFunction,
    dataset: LabelledDataset,
    partition: Optional[str] = None,
    cv: BaseCrossValidator = LeaveOneGroupOut(),
    scoring: Union[str, List[str], Dict[str, ScoreFunction]] = "accuracy",
    data_fn: DataFunction = create_tf_dataset_ragged,
    sample_weight=None,
    log_dir: Optional[Path] = None,
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
    groups = None if partition is None else dataset.get_group_indices(partition)
    n_folds = cv.get_n_splits(dataset.x, dataset.y, groups)
    for fold, (train, test) in enumerate(cv.split(dataset.x, dataset.y, groups)):
        if fit_params.get("verbose", False):
            print(f"\tFold {fold + 1}/{n_folds}")

        x_train = dataset.x[train]
        y_train = dataset.y[train]
        x_test = dataset.x[test]
        y_test = dataset.y[test]
        sw_train = sample_weight[train] if sample_weight else None
        sw_test = sample_weight[test] if sample_weight else None

        callbacks = []
        if log_dir is not None:
            tb_log_dir = log_dir / str(fold + 1)
            callbacks.append(
                TensorBoard(
                    log_dir=tb_log_dir,
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
            **fit_params,
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
