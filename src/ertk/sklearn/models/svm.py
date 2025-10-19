"""Support Vector Machine (SVM) models for scikit-learn."""

from functools import partial
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import kernel_metrics
from sklearn.svm import SVC

__all__ = ["PrecomputedSVC"]


def _get_kernel_func(
    kernel: str,
    degree: int = 3,
    gamma: str | float | None = "auto",
    coef0: float = 0.0,
):
    """Get the kernel function, with parameters, to use in fit() and
    predict(). This is calculated at runtime in order to more easily
    handle changes in parameters such as kernel, gamma, etc.
    """
    f = kernel_metrics()[kernel]
    params = {}
    gamma = None if gamma == "auto" else gamma
    if kernel == "poly":
        params = {"degree": degree, "coef0": coef0, "gamma": gamma}
    elif kernel == "rbf":
        params = {"gamma": gamma}
    return partial(f, **params)


class PrecomputedSVC(SVC):
    """Class that wraps scikit-learn's SVC to precompute the kernel
    values in order to speed up training. The kernel parameter is a
    string which is transparently mapped to and from the corresponding
    callable function with the relevant parameters (`degree`, `gamma`,
    `coef0`). All other parameters are passed directly to SVC.
    """

    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="auto",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )
        self.kernel_name = kernel
        self.kernel = _get_kernel_func(
            self.kernel_name, degree=self.degree, gamma=self.gamma, coef0=self.coef0
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params["kernel"] = self.kernel_name
        return params

    def set_params(self, **params) -> BaseEstimator:
        super().set_params(**params)
        if "kernel" in params:
            self.kernel_name = params["kernel"]
        self.kernel = _get_kernel_func(
            self.kernel_name, degree=self.degree, gamma=self.gamma, coef0=self.coef0
        )
        return self
