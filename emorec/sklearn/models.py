from functools import partial
from typing import Any, Callable, Dict, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC

KernelFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _linear(x, y) -> np.ndarray:
    return np.matmul(x, y.T)


def _poly(x, y, d=2, r=0, gamma: Union[str, float] = 'auto') -> np.ndarray:
    a = np.matmul(x, y.T)
    if gamma == 'auto':
        gamma = 1 / x.shape[1]
    return (gamma * a + r)**d


def _rbf(x, y, gamma: Union[str, float] = 'auto') -> np.ndarray:
    a = np.matmul(x, y.T)
    xx = np.sum(x**2, axis=1)
    yy = np.sum(y**2, axis=1)
    s = xx[:, np.newaxis] + yy[np.newaxis, :]
    if gamma == 'auto':
        gamma = 1 / x.shape[1]
    # <x - y, x - y> = <x, x> + <y, y> - 2<x, y>
    return np.exp(-gamma * (s - 2 * a))


class PrecomputedSVC(SVC):
    """Class that wraps scikit-learn's SVC to precompute the kernel
    values in order to speed up training. The kernel parameter is a
    string which is transparently mapped to and from the corresponding
    callable function with the relevant parameters (degree, gamma,
    coef0). All other parameters are passed directly to SVC.
    """
    KERNELS = {'rbf': _rbf, 'poly': _poly, 'linear': _linear}

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=1e-3, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', break_ties=False,
                 random_state=None):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties, random_state=random_state
        )
        self.kernel_name = kernel
        self.kernel = self._get_kernel_func()

    def get_params(self, deep) -> Dict[str, Any]:
        params = super().get_params(deep)
        params['kernel'] = self.kernel_name
        return params

    def set_params(self, **params) -> BaseEstimator:
        super().set_params(**params)
        if 'kernel' in params:
            self.kernel_name = params['kernel']
        self.kernel = self._get_kernel_func()
        return self

    def _get_kernel_func(self) -> KernelFunction:
        """Get the kernel function, with parameters, to use in fit() and
        predict(). This is calculated at runtime in order to more easily
        handle changes in parameters such as kernel, gamma, etc.
        """
        f = self.KERNELS[self.kernel_name]
        params = {}
        if self.kernel_name == 'poly':
            params = {'d': self.degree, 'r': self.coef0, 'gamma': self.gamma}
        elif self.kernel_name == 'rbf':
            params = {'gamma': self.gamma}
        return partial(f, **params)
