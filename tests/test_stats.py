import numpy as np

from phenotopo import benjamini_hochberg, bootstrap_ratio, group_connectivity, knn_graph, permutation_test
from phenotopo.data import synthetic_cohort


def _setup():
    emb, labels, d = synthetic_cohort(n=600, seed=0)
    return knn_graph(distance=d, k=15), labels


def test_bh_is_monotone_and_bounded():
    q = benjamini_hochberg(np.array([0.001, 0.02, 0.5, 0.9]))
    assert np.all((q >= 0) & (q <= 1))
    assert q[0] <= q[1] <= q[2] <= q[3]


def test_permutation_null_centres_on_one():
    adj, labels = _setup()
    res = permutation_test(adj, labels, n_perm=200, random_state=0)
    off = res["null_mean"].to_numpy()[~np.eye(5, dtype=bool)]
    assert np.all(np.abs(off - 1.0) < 0.25)


def test_permutation_recovers_designed_structure():
    adj, labels = _setup()
    res = permutation_test(adj, labels, n_perm=300, random_state=1)
    pairs = res["pairs"].set_index(["group_a", "group_b"])
    def row(a, b):
        return pairs.loc[(a, b)] if (a, b) in pairs.index else pairs.loc[(b, a)]
    blend = row("Neuro", "Devel"); island = row("Neuro", "Bone")
    assert blend["direction"] == "enriched" and blend["significant"]
    assert island["direction"] == "depleted" and island["significant"]
    assert blend["ratio"] > blend["null_hi95"]
    assert island["ratio"] < island["null_lo95"]
    assert res["ratio"].equals(group_connectivity(adj, labels)["ratio"])


def test_permutation_is_deterministic():
    adj, labels = _setup()
    a = permutation_test(adj, labels, n_perm=50, random_state=3)["pairs"]
    b = permutation_test(adj, labels, n_perm=50, random_state=3)["pairs"]
    assert a.equals(b)


def test_bootstrap_interval_contains_estimate_for_large_groups():
    adj, labels = _setup()
    point = group_connectivity(adj, labels)["ratio"]
    ci = bootstrap_ratio(adj, labels, n_boot=40, random_state=0)
    for g in ["Neuro", "Devel", "Bone"]:
        assert ci["ci_lo95"].loc[g, g] <= point.loc[g, g] * 1.05
        assert ci["ci_hi95"].loc[g, g] >= point.loc[g, g] * 0.95
