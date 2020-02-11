import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from .dataset import ArffDataset

__all__ = [
    'test_one_vs_rest',
    'test_classifier',
    'grid_classifier',
    'print_results',
    'PrecomputedSVC',
    'record_metrics'
]

METRICS = ['prec', 'rec', 'uap', 'uar', 'war']


def linear_kernel(x, y):
    return np.matmul(x, y.T)


def poly_kernel(x, y, d=2, r=0):
    M = linear_kernel(x, y)
    gamma = 1 / x.shape[1]
    return (gamma * M + r)**d


def rbf_kernel(x, y, gamma='scale'):
    M = linear_kernel(x, y)
    xx = np.sum(x**2, axis=1)
    yy = np.sum(y**2, axis=1)
    D = xx[:, np.newaxis] + yy[np.newaxis, :]
    if gamma == 'scale':
        gamma = 1 / x.shape[1]
    return np.exp(-gamma * (D - 2 * M))


class Classifier():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class PrecomputedSVC(SVC):
    def __init__(self, C=1, kernel='rbf', degree=3, gamma='scale', coef0=0,
                 class_weight=None, max_iter=-1):
        if kernel == 'linear':
            self.kernel_func = linear_kernel
        elif kernel == 'poly':
            self.kernel_func = partial(poly_kernel, d=degree, r=coef0)
        elif kernel == 'rbf':
            self.kernel_func = partial(rbf_kernel, gamma=gamma)
        else:
            raise ValueError(
                "kernel must be in {{'linear', 'poly', 'rbf'}}, got '{}'"
                .format(kernel))
        self.kernel_name = kernel

        super().__init__(C=C, kernel='precomputed', degree=degree, gamma=gamma,
                         coef0=coef0, class_weight=class_weight,
                         max_iter=max_iter)

    @property
    def _pairwise(self):
        return False

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['kernel'] = self.kernel_name
        # params = {
        #     'C': self.C,
        #     'kernel': self.kernel_name,
        #     'degree': self.degree,
        #     'gamma': self.gamma,
        #     'coef0': self.coef0,
        #     'class_weight': self.class_weight
        # }
        return params

    def set_params(self, **params):
        if not params:
            return self
        kernel = params.get('kernel', self.kernel_name)
        degree = params.get('degree', self.degree)
        coef0 = params.get('coef0', self.coef0)
        gamma = params.get('gamma', self.gamma)

        if kernel == 'linear':
            self.kernel_func = linear_kernel
        elif kernel == 'poly':
            self.kernel_func = partial(poly_kernel, d=degree, r=coef0)
        elif kernel == 'rbf':
            self.kernel_func = partial(rbf_kernel, gamma=gamma)
        else:
            raise ValueError(
                "kernel must be in {{'linear', 'poly', 'rbf'}}, got '{}'"
                .format(kernel))

        return super().set_params(**params)

    def fit(self, X, y, sample_weight=None):
        self.x_train = X
        K = self.kernel_func(X, X)
        return super().fit(K, y, sample_weight=sample_weight)

    def predict(self, X):
        K = self.kernel_func(X, self.x_train)
        return super().predict(K)


def grid_classifier(params, klass, score_fn, x_train, y_train, x_valid,
                    y_valid):
    classifier = klass(**params)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_valid)
    score = score_fn(y_valid, y_pred)
    return classifier, score


def cross_validate(param_grid, klass, score_fn, x_train, y_train, x_valid,
                   y_valid):
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity())) as pool:
        max_score = 0
        func = partial(grid_classifier, klass=klass, score_fn=score_fn,
                       x_train=x_train, y_train=y_train, x_valid=x_valid,
                       y_valid=y_valid)
        for clf, score in pool.map(func, param_grid):
            if score > max_score:
                max_score = score
                classifier = clf
    return classifier


def record_metrics(df, fold, rep, y_true, y_pred, n_classes):
    df.loc[fold, ('war', 'all', slice(None), rep)] = recall_score(
        y_true, y_pred, average='micro')
    df.loc[fold, ('uar', 'all', slice(None), rep)] = recall_score(
        y_true, y_pred, average='macro')
    df.loc[fold, ('uap', 'all', slice(None), rep)] = precision_score(
        y_true, y_pred, average='macro')
    df.loc[fold, ('rec', 'all', slice(None), rep)] = recall_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)))
    df.loc[fold, ('prec', 'all', slice(None), rep)] = precision_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)))


def print_results(df: pd.DataFrame):
    genders = df.axes[1].get_level_values('gender').unique()
    metrics = df.axes[1].get_level_values('metric').unique()
    labels = df.axes[1].get_level_values('class').unique()
    for gender in genders:
        print("Metrics: mean +- std. dev. over folds")
        print("Across reps:")
        print('                ' + ' '.join(['{:<12}'.format(c) for c in labels]))
        for metric in metrics:
            print('{:<4s} ({}) {}'.format(metric, gender, ' '.join(['{:<4.2f} +- {:<4.2f}'.format(
                  df[(metric, gender, c)].mean().mean(),
                  df[(metric, gender, c)].std().mean()) for c in labels])))
        print()
        print("Across classes and reps:")
        for metric in metrics:
            print('{:<4s}: {:.3f} +- {:.2f} ({:.2f})'.format(
                metric.upper(),
                df[(metric, gender)].mean().mean(),
                df[(metric, gender)].std().mean(),
                df[(metric, gender)].max().max()))
        print("")
        print()


def test_one_vs_rest(classifier_fn,
                     dataset: ArffDataset,
                     gendered=False,
                     reps=1,
                     param_grid=None,
                     splitter=KFold(10)) -> pd.DataFrame:
    genders = ['all', 'f', 'm'] if gendered else ['all']
    labels = sorted([x[:3] for x in dataset.classes])

    rec = pd.DataFrame(
        index=pd.RangeIndex(splitter.get_n_splits(dataset.x, dataset.labels[0], dataset.speaker_indices)),
        columns=pd.MultiIndex.from_product(
            [['prec', 'rec'], genders, labels, list(range(reps))],
            names=['metric', 'gender', 'class', 'rep']))
    for gender in genders:
        groups = dataset.speaker_indices[dataset.gender_indices[gender]]
        x = dataset.x[dataset.gender_indices[gender]]
        for idx in range(dataset.n_classes):
            y = dataset.labels[idx][dataset.gender_indices[gender]]
            for rep in range(reps):
                for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
                    x_train, y_train = x[train], y[train]
                    x_test, y_test = x[test], y[test]

                    if param_grid:
                        classifier = cross_validate(param_grid, classifier_fn,
                                                    recall_score, x_train,
                                                    y_train, x_test, y_test)
                    else:
                        classifier = classifier_fn()

                    y_pred = classifier.predict(x_test)
                    rec[('prec', gender, labels[idx], rep)][fold] = precision_score(y_test, y_pred)
                    rec[('rec', gender, labels[idx], rep)][fold] = recall_score(y_test, y_pred)
    return rec


def test_classifier(classifier_fn,
                    dataset: ArffDataset,
                    mode='all',
                    gendered=False,
                    reps=1,
                    param_grid=None,
                    splitter=KFold(10)) -> pd.DataFrame:
    if mode == 'all':
        labels = sorted([x[:3] for x in dataset.classes])
    else:
        labels = ['neg', 'pos']

    genders = ['all', 'f', 'm'] if gendered else ['all']

    df = pd.DataFrame(
        index=pd.RangeIndex(splitter.get_n_splits(
            dataset.x, dataset.labels[mode], dataset.speaker_indices)),
        columns=pd.MultiIndex.from_product(
            [METRICS, genders, labels, range(reps)],
            names=['metric', 'gender', 'class', 'rep']))
    for gender in genders:
        groups = dataset.speaker_indices[dataset.gender_indices[gender]]
        x = dataset.x[dataset.gender_indices[gender]]
        y = dataset.labels[mode]
        for rep in range(reps):
            for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
                x_train, y_train = x[train], y[train]
                x_test, y_test = x[test], y[test]

                if param_grid:
                    score_fn = partial(recall_score, average='macro')
                    classifier = cross_validate(param_grid, classifier_fn,
                                                score_fn, x_train, y_train,
                                                x_test, y_test)
                else:
                    classifier = classifier_fn()

                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)

                record_metrics(df, fold, rep, y_test, y_pred, len(labels))
    return df
