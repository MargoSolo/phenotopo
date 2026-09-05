import matplotlib
matplotlib.use("Agg")
import numpy as np

from phenotopo import connectivity_robustness, plot_forest
from phenotopo.data import synthetic_cohort


def _result():
    emb, labels, d = synthetic_cohort(n=600, seed=0)
    return connectivity_robustness({"euclid": d, "scaled": d * 3.0}, labels, ks=(8, 15),
                                   min_size=40, n_perm=100, n_boot=30, random_state=0)


def test_protocol_shapes_and_verdicts():
    res = _result()
    assert set(res["groups"]) == {"Neuro", "Devel", "Bone", "Skin", "Renal"}
    assert len(res["configs"]) == 4
    assert len(res["table"]) == 4 * 15          # 10 pairs + 5 self, per config
    s = res["summary"]
    assert s.loc["Neuro – Bone", "verdict"] == "separation"     # designed island
    assert s.loc["Bone (self)", "verdict"] == "cohesive"
    assert set(s["verdict"]) <= {"blend", "separation", "cohesive", "diffuse", "not robust"}


def test_small_groups_are_dropped():
    import pytest
    emb, labels, d = synthetic_cohort(n=600, seed=0)
    res = connectivity_robustness({"euclid": d}, labels, ks=(10,), min_size=100, n_perm=20, n_boot=5)
    assert set(res["groups"]) == {"Neuro", "Devel"}          # Bone (90), Skin, Renal dropped
    assert all(np.sum(labels == g) >= 100 for g in res["groups"])
    with pytest.raises(ValueError):                           # only one group left -> refuse
        connectivity_robustness({"euclid": d}, labels, ks=(10,), min_size=200, n_perm=5, n_boot=2)


def test_forest_plot_runs():
    ax = plot_forest(_result())
    assert ax is not None
