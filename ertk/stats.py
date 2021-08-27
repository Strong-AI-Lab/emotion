from functools import partial
from typing import Callable, List, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

Matrix = List[List[float]]


def _get_dist_func(metric: Union[Callable, str], **kwargs):
    if callable(metric):
        return partial(metric, **kwargs)
    else:
        if metric != "minkowski" and "p" in kwargs:
            del kwargs["p"]
        if metric != "mahalanobis" and "VI" in kwargs:
            del kwargs["VI"]
        return partial(pairwise_distances, metric=metric, **kwargs)


def bhattacharyya_dist(
    mus: Tuple[np.ndarray, np.ndarray], covs: Tuple[np.ndarray, np.ndarray]
):
    mu1 = np.expand_dims(mus[0], 1)
    mu2 = np.expand_dims(mus[1], 1)
    cov1 = np.array(covs[0])
    cov2 = np.array(covs[1])
    cov = (cov1 + cov2) / 2
    sign1, ldet1 = np.linalg.slogdet(cov1)
    sign2, ldet2 = np.linalg.slogdet(cov2)
    _, ldet = np.linalg.slogdet(cov)
    if sign1 <= 0:
        raise ValueError("cov1 is not positive-definite.")
    if sign2 <= 0:
        raise ValueError("cov2 is not positive-definite.")
    covinv = np.linalg.inv(cov)
    db = (mu1 - mu2).T.dot(covinv).dot(mu1 - mu2) / 8 + ldet / 2 - ldet1 / 4 - ldet2 / 4

    return db.item()


def corr_ratio(x: np.ndarray, groups: Union[List[int], np.ndarray]):
    """Calculates correlation ratio for each feature using the given
    groups.

    Args:
    -----
    data: numpy.ndarray
        Data matrix, with shape (n_instances, n_features).
    groups: list or numpy.ndarray
        1D array of groups assignments of length n_instances. Groups
        should be labelled from 0 to G - 1 inclusive, where G is the
        number of groups.

    Returns:
    --------
    eta: numpy.ndarray
        1D array of correlation coefficients of length n_features. Each
        value is in [0, 1] except if a feature takes only one value, in
        which case eta will be nan.
    """
    groups = np.array(groups)
    n_groups = groups.max() + 1
    counts = np.bincount(groups)
    mean = x.mean(0)
    g_means = np.empty((n_groups, x.shape[1]))
    for g in range(n_groups):
        g_means[g, :] = x[groups == g].mean(0)
    num = np.sum(counts[:, None] * (g_means - mean) ** 2, axis=0)
    den = np.sum((x - mean) ** 2, axis=0)
    old_err = np.seterr(divide="ignore", invalid="ignore")
    eta2 = num / den
    np.seterr(**old_err)
    return np.sqrt(eta2)


def dunn(
    x: np.ndarray,
    clusters: Union[List[int], np.ndarray],
    intra_method: str = "mean",
    inter_method: str = "cent",
    metric: Union[Callable, str] = "l2",
    p: int = 2,
):
    """Calculates the Dunn index for cluster "goodness".

    Args:
    -----
    data: numpy.ndarray
        Data matrix, with shape (n_instances, n_features).
    clusters: list or numpy.ndarray
        1D array of cluster assignments of length n_instances. Clusters
        should be labelled from 0 to C - 1 inclusive, where C is the
        number of clusters.
    intra_method: str
        Method for calculating intra-cluster distance. One of "max",
        "mean", "cent".
    inter_method: str
        Method for calculating inter-cluster distance. One of "cent".
    metric: str or callable
        Distance metric. If str, must be one of the sklearn or scipy
        distance methods. If callable, must take one positional argument
        and return a pairwise distance matrix.
    p: int
        Value of p for p-norm when using "lp" distance metric.

    Returns:
    --------
    dunn: float
        The Dunn index for this data and cluster assignment.
    """
    clusters = np.array(clusters, dtype=int)
    n_clusters = clusters.max() + 1
    d = _get_dist_func(metric, p=p)

    intra = np.zeros(n_clusters)
    for c in range(n_clusters):
        clust_data = x[clusters == c]
        if intra_method == "max":
            idx = np.triu_indices(len(clust_data))
            intra[c] = d(clust_data)[idx].max()
        elif intra_method == "mean":
            idx = np.triu_indices(len(clust_data))
            intra[c] = d(clust_data)[idx].mean()
        elif intra_method == "cent":
            mean = clust_data.mean(0)
            intra[c] = d(clust_data, mean[None, :]).mean()

    inter = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        inter[i, i] = np.inf  # To avoid min = 0
        for j in range(i + 1, n_clusters):
            if inter_method == "cent":
                mean_i = x[clusters == i].mean(0)
                mean_j = x[clusters == j].mean(0)
                inter[i, j] = inter[j, i] = d(mean_i[None, :], mean_j[None, :])

    return inter.min() / intra.max()


def kappa(data: np.ndarray):
    """Calculates Fleiss' kappa for inter-rater agreement.

    Args:
    -----
    data: numpy.ndarray
        The data matrix, in the form (raters x units).
    """
    cats = np.unique(data)
    n, N = data.shape

    counts = np.stack([np.sum(data == c, 0) for c in cats], 1)

    p_j = np.sum(counts, axis=0) / (N * n)
    assert np.isclose(np.sum(p_j), 1)
    Pe = np.sum(p_j ** 2)

    P = (np.sum(counts ** 2, 1) - n) / (n * (n - 1))
    Pbar = np.mean(P)

    return (Pbar - Pe) / (1 - Pe)


class Deltas:
    @staticmethod
    def nominal(c: int, k: int):
        return float(c != k)

    @staticmethod
    def interval(c: float, k: float):
        return (c - k) ** 2


def alpha(
    data: np.ndarray,
    delta: Union[Callable[[int, int], float], List[List[float]], str] = "nominal",
):
    """Calculates Krippendorff's alpha coefficient [1, sec. 11.3] for
    inter-rater agreement.

    [1] K. Krippendorff, Content analysis: An introduction to its
    methodology. Sage publications, 2004.

    Args:
    -----
    data: numpy.ndarray
        The data matrix, shape (n_raters, n_units). Each cell (i, j)
        represents the value assigned to unit j by rater i, or 0
        representing no response.
    delta: callable, 2-D array-like or str
        The delta metric. Default is the nominal metric, which takes the
        value 1 in case c != k and 0 otherwise.
    """
    # The following implementation was based off the Wikipedia article:
    # https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

    # Response categories go from 1 to R, 0 represents no response
    R = np.max(data)
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=R + 1), 0, data).T
    count_sum = np.sum(counts, 0)
    assert len(count_sum) == R + 1

    def ordinal(c: int, k: int):
        if k < c:
            c, k = k, c
        s = (
            sum(count_sum[g] for g in range(c, k + 1))
            - (count_sum[c] + count_sum[k]) / 2
        )
        return s ** 2

    if isinstance(delta, str):
        delta = {
            "nominal": Deltas.nominal,
            "ordinal": ordinal,
            "interval": Deltas.interval,
        }[delta]

    if not callable(delta):
        try:
            delta[0][0]
        except IndexError:
            raise TypeError("delta must be either str, callable or 2D array.")

        def _delta(c, k):
            new_delta = delta
            return new_delta[c][k]

        delta = _delta

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
