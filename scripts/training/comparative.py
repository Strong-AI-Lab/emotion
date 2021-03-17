import importlib
import json
import os
from functools import partial, wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Type, Union

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from click_option_group import optgroup
from emorec.dataset import LabelledDataset
from emorec.sklearn.models import PrecomputedSVC
from emorec.tensorflow.classification import tf_cross_validate
from emorec.tensorflow.models.zhang2019 import create_windowed_dataset
from emorec.tensorflow.utils import create_tf_dataset_ragged
from emorec.utils import PathlibPath
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (get_scorer, make_scorer, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     LeaveOneGroupOut, cross_validate)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Adam, Optimizer


# SVM classifiers
def get_svm_params(kind='linear') -> Dict[str, Sequence[object]]:
    param_grid = {'C': 2.0**np.arange(-6, 7, 2)}
    if kind == 'linear':
        param_grid.update({'kernel': ['poly'], 'degree': [1], 'coef0': [0]})
    elif kind == 'poly2':
        param_grid.update({'kernel': ['poly'], 'degree': [2]})
        param_grid['coef0'] = [-1, 0, 1]
    elif kind == 'poly3':
        param_grid.update({'kernel': ['poly'], 'degree': [3]})
        param_grid['coef0'] = [-1, 0, 1]
    elif kind == 'rbf':
        param_grid['kernel'] = ['rbf']
        param_grid['gamma'] = 2.0**np.arange(-12, -1, 2)
    else:
        raise NotImplementedError("Other kinds of SVM are not currently "
                                  "implemented.")
    return param_grid


# Random forest classifiers
def get_rf_params() -> Dict[str, Sequence[object]]:
    return {'n_estimators': [100, 250, 500], 'max_depth': [None, 10, 20, 50]}


def compile(model_fn: Callable[..., Model],
            opt_cls: Type[Optimizer] = Adam,
            opt_params: Dict[str, object] = dict(learning_rate=0.0001),
            metrics: List[Union[str, Metric]] = [
                'sparse_categorical_accuracy'],
            loss: Union[str, Loss] = 'sparse_categorical_crossentropy',
            **compile_args):
    @wraps(model_fn)
    def f(*args, **kwargs):
        model = model_fn(*args, **kwargs)
        model.compile(optimizer=opt_cls(**opt_params), metrics=metrics,
                      loss=loss, **compile_args)
        return model
    return f


# Fully connected feedforward networks.
def dense_keras_model(n_features: int, n_classes: int, layers: int = 1) \
        -> Model:
    """Creates a Keras model with hidden layers and ReLU activation,
    with 50% dropout.
    """
    inputs = Input((n_features,), name='input')
    x = inputs
    for _ in range(layers):
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)


# Arbitrary TF models
def get_tf_model(name: str, n_features: int, n_classes: int) -> Model:
    module = importlib.import_module(f'emorec.tensorflow.models.{name}')
    model_fn = getattr(module, 'model')
    if n_features > 1:
        model = model_fn(n_features, n_classes)
    else:
        model = model_fn(n_classes)
    return model


def test_classifier(kind: str,
                    dataset: LabelledDataset,
                    reps: int = 1,
                    results: Optional[Path] = None,
                    logs: Optional[Path] = None,
                    verbose: bool = False,
                    lr: float = 1e-4,
                    epochs: int = 50,
                    bs: int = 64):
    splitter = LeaveOneGroupOut()
    if len(dataset.speakers) > 12:
        splitter = GroupKFold(6)

    class_weight = (dataset.n_instances
                    / (dataset.n_classes * dataset.class_counts))
    sample_weight = class_weight[dataset.y]

    metrics = (['uar', 'war'] + [x + '_rec' for x in dataset.classes]
               + [x + '_prec' for x in dataset.classes])
    df = pd.DataFrame(index=pd.RangeIndex(1, reps + 1, name='rep'),
                      columns=metrics + ['params'])

    scoring = {'war': get_scorer('accuracy'),
               'uar': get_scorer('balanced_accuracy')}
    for i, c in enumerate(dataset.classes):
        scoring.update({
            c + '_rec': make_scorer(recall_score, average=None, labels=[i]),
            c + '_prec': make_scorer(precision_score, average=None, labels=[i])
        })

    for rep in range(1, reps + 1):
        print(f"Rep {rep}/{reps}")
        params = {}
        if kind.startswith(('svm/', 'mlp/')) or kind == 'rf':
            if kind.startswith('mlp'):
                # Force CPU only to do in parallel, supress TF errors
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                params.update(dict(learning_rate=lr, batch_size=bs,
                                   epochs=epochs))
                layers = 1
                _layers = kind.split('/')[-1]
                if _layers == '2layer':
                    layers = 2
                elif _layers == '3layer':
                    layers = 3
                model_fn = compile(dense_keras_model,
                                   opt_params=dict(learning_rate=lr))
                clf = KerasClassifier(
                    model_fn, n_features=dataset.n_features,
                    n_classes=dataset.n_classes, layers=layers, batch_size=bs,
                    epochs=epochs, verbose=False
                )
            else:
                if kind.startswith('svm'):
                    param_grid = get_svm_params(kind.split('/')[-1])
                    _clf = PrecomputedSVC()
                else:
                    param_grid = get_rf_params()
                    _clf = RandomForestClassifier()
                clf = GridSearchCV(_clf, param_grid, cv=splitter,
                                   scoring='balanced_accuracy', n_jobs=-1)
                # Get best hyperparameters through inner CV
                clf.fit(
                    dataset.x, dataset.y, groups=dataset.speaker_group_indices,
                    sample_weight=sample_weight
                )
                params.update(clf.best_params_)
                clf = clf.best_estimator_
            fit_params = dict(sample_weight=sample_weight)
            scores = cross_validate(
                clf, dataset.x, dataset.y, cv=splitter, scoring=scoring,
                groups=dataset.speaker_group_indices,
                fit_params=fit_params, n_jobs=-1, verbose=int(verbose)
            )
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
            data_fn = create_tf_dataset_ragged
            if kind == 'zhang2019':
                data_fn = create_windowed_dataset
            data_fn = partial(data_fn, batch_size=bs)
            params.update(dict(learning_rate=lr, batch_size=bs, epochs=epochs))

            model_fn = partial(
                compile(get_tf_model, opt_params={'learning_rate': lr}),
                kind, dataset.n_features, dataset.n_classes
            )
            if rep == 1:
                # To print model params
                _model = model_fn()
                _model.summary()
                del _model
            tf.keras.backend.clear_session()

            scores = tf_cross_validate(
                model_fn, dataset.x, dataset.y, cv=splitter, scoring=scoring,
                groups=dataset.speaker_group_indices, data_fn=data_fn,
                sample_weight=sample_weight, log_dir=None,
                fit_params=dict(epochs=epochs, verbose=verbose)
            )
            if logs:
                log_dir = logs / ('rep_' + str(rep))
                log_dir.mkdir(parents=True, exist_ok=True)
                log_df = pd.DataFrame.from_dict({
                    # e.g. (0, 'loss'): 1.431...
                    (fold, key): val
                    for fold in range(len(scores['history']))
                    for key, val in scores['history'][fold].items()
                })
                log_df.to_csv(log_dir / 'history.csv', header=True, index=True)
        # Make string to add to final dataframe
        params_s = json.dumps(params)

        mean_scores = {k[5:]: np.mean(v) for k, v in scores.items()
                       if k.startswith('test_')}
        war = mean_scores['war']
        uar = mean_scores['uar']
        recall = tuple(mean_scores[c + '_rec'] for c in dataset.classes)
        precision = tuple(mean_scores[c + '_prec'] for c in dataset.classes)

        df.loc[rep, 'params'] = params_s
        df.loc[rep, 'war'] = war
        df.loc[rep, 'uar'] = uar
        for i, c in enumerate(dataset.classes):
            df.loc[rep, c + '_rec'] = recall[i]
            df.loc[rep, c + '_prec'] = precision[i]

    if results:
        results.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results)
        print(f"Wrote CSV to {results}")
    else:
        print(df.to_string())


@click.command()
@click.argument('kind')
@click.argument('input', type=PathlibPath(exists=True, dir_okay=False))
@click.argument('labels', type=PathlibPath(exists=True, dir_okay=False))
@optgroup.group('Dataset options')
@optgroup.option('--pad', type=int,
                 help="Optionally pad input sequences to this length.")
@optgroup.option(
    '--clip', type=int,
    help="Optionally clip input sequences to this length, after any padding."
)
@optgroup.group('Results options')
@optgroup.option('--results', type=Path, help="Results directory.")
@optgroup.group('Cross-validation options')
@optgroup.option('--reps', type=int, default=1, show_default=True,
                 help="The number of repetitions to do per test.")
@optgroup.group('Misc. options')
@optgroup.option('--verbose', is_flag=True, help="Verbose training.")
@optgroup.option('--logs', type=Path,
                 help="Folder to write training logs per fold.")
@optgroup.group('Model-specific options')
@optgroup.option('--learning_rate', type=float, default=1e-4,
                 show_default=True)
@optgroup.option('--batch_size', type=int, default=64, show_default=True)
@optgroup.option('--epochs', type=int, default=50, show_default=True)
def main(kind: str, input: Path, labels: Path, pad: int, clip: int,
         results: Path, reps: int, verbose: bool, logs: Path,
         learning_rate: float, batch_size: int, epochs: int):
    """Runs a training routine using the given classifier KIND and INPUT
    data. Metrics are optionally written to a results file.
    """

    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    speaker_path = labels.parent / 'speaker.csv'
    dataset = LabelledDataset(input, labels, speaker_path=speaker_path)
    print(dataset)
    dataset.normalise(normaliser=StandardScaler(), scheme='speaker')
    if pad:
        dataset.pad_arrays(pad)
    if clip:
        dataset.clip_arrays(clip)

    test_classifier(
        kind, dataset, reps=reps, results=results, logs=logs, verbose=verbose,
        lr=learning_rate, epochs=epochs, bs=batch_size
    )


if __name__ == "__main__":
    main()
