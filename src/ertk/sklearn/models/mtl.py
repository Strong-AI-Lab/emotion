import warnings

import liblinear.liblinearutil as liblinearutil
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

_EPS = 1e-15


class BaseMTFL(BaseEstimator):
    """Linear regressor based on MTFL. Code adapted from
    https://github.com/argyriou/multi_task_learning

    Parameters
    ----------
    gamma: float
        Gamma value.
    eps_init: float
        Initial value for epsilon.
    n_iters: int
        Number of iterations per step.
    """

    def __init__(
        self,
        gamma: float = 1,
        epsilon: float = 1e-4,
        n_iters: int = 10,
        loss: str = "logistic",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.loss = loss

    def _loss(self, X, y, task_idxs):
        num_tasks = len(task_idxs)
        W = np.zeros((X.shape[1], num_tasks))
        cost = 0
        err = 0
        reg = 0
        for t_id in range(num_tasks):
            task_x = X[task_idxs[t_id]]
            task_y = y[task_idxs[t_id]]

            if self.loss == "logistic":
                model = liblinearutil.train(
                    task_y, task_x, f"-s 0 -e 1e-3 -c {self.gamma} -q"
                )
                w, b = model.get_decfun()
            elif self.loss == "svm":
                model = liblinearutil.train(
                    task_y, task_x, f"-s 2 -e 1e-3 -c {self.gamma} -q"
                )
                w, b = model.get_decfun()
            elif self.loss == "ols":
                K = task_x.T @ task_x
                mat_inv = np.linalg.inv(K + self.gamma * np.eye(K.shape[0]))
                w = mat_inv @ task_x.T @ task_y

                # K = task_x @ task_x.T  # Linear kernel
                # mat_inv = np.linalg.inv(K + self.gamma * np.eye(K.shape[0]))
                # a = mat_inv @ task_x.T @ task_y
                # cost += self.gamma * task_y @ a
                # err += self.gamma**2 * a.T @ a
                # W[:, t_id] = task_x.T @ a
            w = np.asarray(w)
            W[:, t_id] = w
            _cost = self.gamma * w.T @ w
            cost += _cost
        reg = cost - err
        return W, cost, err, reg

    def Dmin(self, a):
        return a / a.sum()

    def Dmin_eps(self, a, eps):
        return self.Dmin(np.sqrt(a ** 2 + eps))

    def fit(self, X, y, tasks, sample_weight=None):
        self._fit_feat(X, y, tasks)
        return self

    def _fit_feat(self, X, y, tasks):
        task_ids = np.unique(tasks)
        task_idxs = [np.flatnonzero(tasks == t) for t in task_ids]
        dim = X.shape[1]
        Dini = np.eye(dim) / dim

        def fmeth(a):
            _a = np.zeros_like(a)
            _a[a > _EPS] = 1 / a[a > _EPS]
            return _a

        # train_alternating
        U, s, _ = np.linalg.svd(Dini, hermitian=True)
        fS = np.sqrt(fmeth(s))
        fS[fS > _EPS] = 1 / fS[fS > _EPS]
        fD_isqrt = U @ np.diag(fS) @ U.T
        _costs = np.empty((self.n_iters, 3))
        for i in range(self.n_iters):
            W, cost, err, reg = self._loss(X @ fD_isqrt, y, task_idxs)
            W = fD_isqrt @ W

            _costs[i] = [cost, err, reg]

            U, s, _ = np.linalg.svd(W)
            if dim > len(task_ids):
                s = np.r_[s, np.zeros(dim - len(task_ids))]
            Smin = self.Dmin_eps(s, eps=self.epsilon)
            D = U @ np.diag(Smin) @ U.T

            U, s, _ = np.linalg.svd(D, hermitian=True)
            fS = np.sqrt(fmeth(s))
            fS[fS > _EPS] = 1 / fS[fS > _EPS]
            fD_isqrt = U @ np.diag(fS) @ U.T
        # train_alternating

        s = np.linalg.svd(D, compute_uv=False)
        _costs[:, [0, 2]] = (
            _costs[:, [0, 2]] + self.gamma * self.epsilon * fmeth(s).sum()
        )
        self.mineps_ = self.epsilon
        self.W_ = W
        self.D_ = D
        self.costs_ = [_costs]

    def predict(self, X):
        return X @ self.W_


class MTFLClassifier(BaseMTFL, ClassifierMixin):
    """Classifier based on MTFL. Code adapted from
    https://github.com/argyriou/multi_task_learning

    Parameters
    ----------
    gamma: float
        Gamma value.
    eps_init: float
        Initial value for epsilon.
    n_iters: int
        Number of iterations per step.
    loss: str
        Loss to use.
    aggregate: str
        Method of aggregation. One of "logistic", "weighted",
        "unweighted".
    """

    def __init__(
        self,
        gamma: float = 1,
        epsilon: float = 1e-4,
        n_iters: int = 10,
        loss: str = "svm",
        aggregate: str = "weighted",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.loss = loss
        self.aggregate = aggregate

    def fit(self, X, y, tasks, sample_weight=None):
        if len(np.unique(y)) > 2:
            raise ValueError("`y` should have only two classes")
        y = y.copy()
        y[y == 0] = -1
        self._fit_feat(X, y, tasks)
        if self.aggregate == "logistic":
            self.logistic_ = LogisticRegression(penalty="none")
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.logistic_.fit(super().predict(X), y)
            warnings.simplefilter("default", ConvergenceWarning)
        return self

    def decision_function(self, X):
        preds = super().predict(X)
        if self.aggregate == "logistic":
            return self.logistic_.decision_function(preds)
        return preds.mean(-1)

    def predict(self, X):
        preds = super().predict(X)
        if self.aggregate == "logistic":
            preds = self.logistic_.predict(preds)
        elif self.aggregate == "weighted":
            preds = np.sign(preds.mean(-1))
        elif self.aggregate == "unweighted":
            preds = np.sign(np.sign(preds).mean(-1))
        preds[preds == -1] = 0
        return preds


class MultiDimensionalCC(ClassifierChain):
    """Classifier chain for multi-dimensional output. Adapted from
    sklearn's ClassifierChain and _BaseChain.

    Parameters
    ----------
    base_estimator: estimator
        The base estimator from which the classifier chain is built.
    order: list of lists
        The order of the outputs in the classifier chain.
    """

    def __init__(
        self,
        base_estimator,
        *,
        order=None,
        cv=None,
        random_state=None,
        verbose: bool = False,
    ) -> None:
        super().__init__(base_estimator, order=order, cv=cv, random_state=random_state)
        self.verbose = verbose

    def fit(self, X, Y, **fit_params):
        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)

        random_state = check_random_state(self.random_state)
        self.order_ = self.order
        if isinstance(self.order_, tuple):
            self.order_ = np.array(self.order_)

        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == "random":
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError("invalid order")

        self.estimators_ = [clone(self.base_estimator) for _ in range(Y.shape[1])]

        encoder = OneHotEncoder(sparse=False)
        Y_onehot = encoder.fit_transform(Y[:, self.order_])
        cat_sizes = np.array([len(x) for x in encoder.categories_])
        cat_sizes_cumsum = np.concatenate([[0], cat_sizes.cumsum()])
        total_size = cat_sizes_cumsum[-1]

        if self.cv is None:
            Y_pred_chain = Y_onehot
            if sp.issparse(X):
                X_aug = sp.hstack((X, Y_pred_chain), format="lil")
                X_aug = X_aug.tocsr()
            else:
                X_aug = np.hstack((X, Y_pred_chain))
        elif sp.issparse(X):
            Y_pred_chain = sp.lil_matrix((X.shape[0], total_size))
            X_aug = sp.hstack((X, Y_pred_chain), format="lil")
        else:
            Y_pred_chain = np.zeros((X.shape[0], total_size))
            X_aug = np.hstack((X, Y_pred_chain))

        del Y_pred_chain

        for chain_idx, estimator in enumerate(self.estimators_):
            y = Y[:, self.order_[chain_idx]]
            estimator.fit(
                X_aug[:, : X.shape[1] + cat_sizes_cumsum[chain_idx]], y, **fit_params
            )
            if self.cv is not None and chain_idx < len(self.estimators_) - 1:
                col_start = X.shape[1] + cat_sizes_cumsum[chain_idx]
                col_end = X.shape[1] + cat_sizes_cumsum[chain_idx + 1]
                cv_result = cross_val_predict(
                    self.base_estimator, X_aug[:, :col_start], y=y, cv=self.cv
                )
                y_oh = OneHotEncoder(sparse=False).fit_transform(
                    cv_result.reshape(-1, 1)
                )
                if sp.issparse(X_aug):
                    X_aug[:, col_start:col_end] = y_oh
                else:
                    X_aug[:, col_start:col_end] = y_oh

        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=True, reset=False)
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))

        X_aug = X
        for chain_idx, estimator in enumerate(self.estimators_):
            if chain_idx != 0:
                y_oh = OneHotEncoder(
                    sparse=False, categories=[self.classes_[chain_idx - 1]]
                ).fit_transform(Y_pred_chain[:, chain_idx - 1].reshape(-1, 1))
                if sp.issparse(X):
                    X_aug = sp.hstack((X_aug, y_oh))
                else:
                    X_aug = np.hstack((X_aug, y_oh))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_pred = Y_pred_chain[:, inv_order]

        return Y_pred
