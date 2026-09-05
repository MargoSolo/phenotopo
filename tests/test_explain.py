import numpy as np
import pytest

from phenotopo import Cohort, Ontology, explain_groups, plot_explain
from phenotopo.data import synthetic_hpo_cohort

import matplotlib
matplotlib.use("Agg")


def _planted():
    """Group A has 'Seizure' (and its child), group B has 'Short stature'."""
    onto = Ontology([("Status epilepticus", "Seizure"), ("Seizure", "Nervous"),
                     ("Short stature", "Skeletal"), ("Nervous", "root"), ("Skeletal", "root")],
                    {t: t for t in ["Status epilepticus", "Seizure", "Nervous",
                                    "Short stature", "Skeletal", "root"]})
    rng = np.random.RandomState(0)
    present, labels = [], []
    for i in range(200):
        if i < 100:
            present.append({"Status epilepticus"} if rng.rand() < 0.9 else {"Short stature"})
            labels.append("A")
        else:
            present.append({"Short stature"} if rng.rand() < 0.9 else {"Status epilepticus"})
            labels.append("B")
    return Cohort(ids=[f"P{i}" for i in range(200)], present=present, ontology=onto), np.array(labels)


def test_planted_difference_is_recovered_with_effect_size_and_ci():
    c, labels = _planted()
    res = explain_groups(c, labels, "A", "B", n_perm=300, min_effect=0.2)
    top = res["top"].set_index("term")
    assert "Status epilepticus" in top.index
    assert top.loc["Status epilepticus", "effect_pp"] > 50
    lo, hi = top.loc["Status epilepticus", ["ci_lo_pp", "ci_hi_pp"]]
    assert lo < top.loc["Status epilepticus", "effect_pp"] < hi and lo > 0
    assert res["method"].startswith("Westfall-Young")


def test_redundant_ancestors_are_pruned_in_favour_of_the_specific_term():
    c, labels = _planted()
    res = explain_groups(c, labels, "A", "B", n_perm=200, min_effect=0.2)
    # 'Seizure' and 'Nervous' carry exactly the same signal as 'Status epilepticus'
    assert "Status epilepticus" in set(res["top"]["term"])
    assert {"Seizure", "Nervous"} & set(res["top"]["term"]) == set()
    pruned = res["table"].set_index("term")["redundant_with"]
    assert pruned["Seizure"] == "Status epilepticus"


def test_effect_size_threshold_actually_filters():
    c, labels = _planted()
    strict = explain_groups(c, labels, "A", "B", n_perm=200, min_effect=0.95)
    assert len(strict["top"]) == 0                       # significant, but no term is that large
    loose = explain_groups(c, labels, "A", "B", n_perm=200, min_effect=0.2)
    assert len(loose["top"]) > 0


def test_no_difference_yields_no_terms():
    rng = np.random.RandomState(1)
    onto = Ontology([("a", "root"), ("b", "root")], {t: t for t in ["a", "b", "root"]})
    present = [{"a"} if rng.rand() < 0.5 else {"b"} for _ in range(200)]
    labels = np.array(["A"] * 100 + ["B"] * 100)
    c = Cohort(ids=[f"P{i}" for i in range(200)], present=present, ontology=onto)
    res = explain_groups(c, labels, "A", "B", n_perm=300, min_effect=0.1)
    assert len(res["top"]) == 0


def test_group_b_defaults_to_the_rest_and_bh_fallback_works():
    c = synthetic_hpo_cohort(n=200, seed=3)
    res = explain_groups(c, c.labels("diagnosis"), "GENE_A", n_perm=0, min_effect=0.15)
    assert res["groups"] == ("GENE_A", "rest")
    assert "Benjamini" in res["method"]
    assert res["sizes"]["a"] + res["sizes"]["b"] == 200


def test_plot_and_input_validation():
    c, labels = _planted()
    res = explain_groups(c, labels, "A", "B", n_perm=100, min_effect=0.2)
    assert plot_explain(res) is not None
    assert plot_explain(explain_groups(c, labels, "A", "B", n_perm=100, min_effect=0.99)) is not None
    with pytest.raises(ValueError):
        explain_groups(c, labels, "nonexistent-group")
    with pytest.raises(ValueError):
        explain_groups(c, labels[:10], "A")
