import numpy as np
import pandas as pd

from phenotopo import group_centroids, group_connectivity, knn_graph
from phenotopo.data import synthetic_cohort


def test_knn_graph_is_symmetric_without_self_loops():
    emb, labels, d = synthetic_cohort(n=200, seed=1)
    a = knn_graph(distance=d, k=5)
    assert a.shape == (200, 200)
    assert (a != a.T).nnz == 0
    assert a.diagonal().sum() == 0
    assert a.nnz > 0


def test_knn_graph_accepts_features():
    emb, labels, _ = synthetic_cohort(n=150, seed=2)
    a = knn_graph(X=emb, k=5)
    assert (a != a.T).nnz == 0


def test_knn_graph_rejects_ambiguous_input():
    emb, _, d = synthetic_cohort(n=50, seed=0)
    try:
        knn_graph(distance=d, X=emb)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_connectivity_matches_designed_structure():
    emb, labels, d = synthetic_cohort(n=900, seed=0)
    conn = group_connectivity(knn_graph(distance=d, k=15), labels)
    ratio = conn["ratio"]
    # Devel is placed on top of Neuro -> strongly connected; Bone is an island
    assert ratio.loc["Neuro", "Devel"] > 1.0
    assert ratio.loc["Neuro", "Bone"] < 0.2
    assert ratio.loc["Bone", "Devel"] < 0.2
    assert set(conn["sizes"].index) == set(labels)
    assert np.allclose(ratio.values, ratio.values.T)


def test_centroids_are_per_group():
    emb, labels, _ = synthetic_cohort(n=300, seed=3)
    c = group_centroids(emb, labels)
    assert isinstance(c, pd.DataFrame)
    assert set(c.index) == set(labels)
    assert list(c.columns) == ["x", "y"]
