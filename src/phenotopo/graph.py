"""k-NN graph over a cohort and PAGA-style connectivity between labelled groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import NearestNeighbors


def knn_graph(
    distance: np.ndarray | None = None,
    k: int = 15,
    X: np.ndarray | None = None,
    metric: str = "euclidean",
) -> csr_matrix:
    """Symmetric unweighted k-nearest-neighbour adjacency matrix.

    Pass either a precomputed square ``distance`` matrix or a feature matrix
    ``X`` (with ``metric``). An edge exists if either point lists the other among
    its ``k`` nearest neighbours; self-edges are removed.
    """
    if (distance is None) == (X is None):
        raise ValueError("pass exactly one of `distance` or `X`")
    if distance is not None:
        d = np.asarray(distance, dtype=float)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError("`distance` must be a square matrix")
        n = d.shape[0]
        d = d.copy()
        np.fill_diagonal(d, np.inf)
        nn = np.argsort(d, axis=1)[:, :k]
    else:
        X = np.asarray(X)
        n = X.shape[0]
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(n), nn.shape[1])
    cols = nn.ravel()
    a = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(float)
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def group_connectivity(adjacency: csr_matrix, labels) -> dict:
    """PAGA-style connectivity between groups.

    For each pair of groups the number of k-NN edges crossing between them is
    compared with the number expected if edges were placed at random while
    preserving every group's total degree (a configuration-model null, the same
    null that underlies modularity). The ratio ``observed / expected`` is the
    connectivity; ``1`` means "as connected as chance", larger means the groups
    genuinely blend into one another, smaller means they are separated.

    This is the abstraction introduced by PAGA (Wolf et al., *Genome Biology*
    2019) for single-cell data, applied here to any labelled cohort.

    Returns
    -------
    dict with
        ``ratio``     DataFrame group × group, observed / expected edges;
        ``observed``  DataFrame of raw crossing-edge counts;
        ``sizes``     Series of group sizes;
        ``labels``    the ordered list of groups.
    """
    labels = np.asarray(labels)
    groups = [g for g in pd.unique(labels)]
    idx = {g: i for i, g in enumerate(groups)}
    member = np.array([idx[g] for g in labels])
    a = adjacency.tocoo()
    m = len(groups)
    observed = np.zeros((m, m), dtype=float)
    np.add.at(observed, (member[a.row], member[a.col]), a.data)
    observed = (observed + observed.T) / 2.0           # each undirected edge counted once per pair
    degree = observed.sum(axis=1)                       # degree mass of each group
    total = degree.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.outer(degree, degree) / total
        ratio = np.where(expected > 0, observed / expected, 0.0)
    sizes = pd.Series({g: int((labels == g).sum()) for g in groups})
    return {
        "ratio": pd.DataFrame(ratio, index=groups, columns=groups),
        "observed": pd.DataFrame(observed, index=groups, columns=groups),
        "sizes": sizes,
        "labels": groups,
    }


def group_centroids(embedding: np.ndarray, labels) -> pd.DataFrame:
    """Median position of every group in a 2-D embedding (median resists outliers)."""
    embedding = np.asarray(embedding)
    labels = np.asarray(labels)
    rows = {g: np.median(embedding[labels == g], axis=0) for g in pd.unique(labels)}
    return pd.DataFrame(rows, index=["x", "y"]).T
