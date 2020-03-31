from typing import Callable, List, Union

import numpy as np

__all__ = ['kappa', 'alpha', 'Deltas']

Matrix = List[List[float]]


def kappa(data: np.ndarray):
    cats = np.unique(data)
    n, N = data.shape

    counts = np.stack([np.sum(data == c, 0) for c in cats], 1)

    p_j = np.sum(counts, axis=0) / (N * n)
    assert np.isclose(np.sum(p_j), 1)
    Pe = np.sum(p_j**2)

    P = (np.sum(counts**2, 1) - n) / (n * (n - 1))
    Pbar = np.mean(P)

    return (Pbar - Pe) / (1 - Pe)


class Deltas:
    @staticmethod
    def nominal(c, k):
        return float(c != k)


def alpha(data: np.ndarray, delta: Union[Callable[[int, int], float], Matrix]):
    def _pad(x):
        return np.pad(x, [(0, R + 1 - x.shape[0])])

    if not callable(delta):
        try:
            delta[0, 0]
        except IndexError:
            raise TypeError("delta must be either callable or 2D array.")

        def _delta(c, k):
            return delta[c, k]
        delta = _delta

    R = np.max(data)

    counts = np.apply_along_axis(lambda x: _pad(np.bincount(x)), 0, data).T
    m_u = np.sum(counts[:, 1:], 1)

    valid = m_u >= 2
    counts = counts[valid]
    m_u = m_u[valid]
    data = data[:, valid]

    n = np.sum(m_u)

    n_cku = np.matmul(counts[:, :, None], counts[:, None, :])
    for i in range(R + 1):
        n_cku[:, i, i] = counts[:, i] * (counts[:, i] - 1)

    D_o = 0
    for c in range(1, R + 1):
        for k in range(1, R + 1):
            D_o += delta(c, k) * n_cku[:, c, k]
    D_o = np.sum(D_o / (n * (m_u - 1)))

    D_e = 0
    P_ck = np.bincount(data.flat)
    for c in range(1, R + 1):
        for k in range(1, R + 1):

                D_e += delta(c, k) * P_ck[c] * P_ck[k]
    D_e /= n * (n - 1)

    return 1 - D_o / D_e
