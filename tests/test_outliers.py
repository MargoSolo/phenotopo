import numpy as np
import pandas as pd
import pytest

from phenotopo import Cohort, Ontology, explain_outlier, patient_outliers
from phenotopo.data import synthetic_cohort, synthetic_hpo_cohort


def test_isolation_and_discordance_are_reported_separately():
    emb, labels, d = synthetic_cohort(n=300, seed=0)
    out = patient_outliers(d, labels, k=10)
    assert len(out) == len(labels)
    assert {"isolation_pct", "discordance", "neighbourhood_majority", "flag"} <= set(out.columns)
    assert out["discordance"].between(0, 1).all()
    assert out["isolation_pct"].max() <= 100
    # a discordant patient's neighbourhood really does disagree with its label
    disc = out[out["flag"].str.contains("discordant")]
    assert (disc["neighbourhood_majority"] != disc["label"]).all()


def test_a_planted_discordant_patient_is_found():
    onto = Ontology([("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B"), ("A", "root"), ("B", "root")],
                    {t: t for t in ["a1", "a2", "b1", "b2", "A", "B", "root"]})
    present = [{"a1", "a2"} for _ in range(30)] + [{"b1", "b2"} for _ in range(30)]
    labels = np.array(["A"] * 30 + ["B"] * 30)
    present[0] = {"b1", "b2"}                                   # A-labelled, B-phenotyped
    c = Cohort(ids=[f"P{i}" for i in range(60)], present=present, ontology=onto)
    out = patient_outliers(c.distance("cosine"), labels, ids=c.ids, k=5).set_index("id")
    assert out.loc["P0", "neighbourhood_majority"] == "B"
    assert "discordant" in out.loc["P0", "flag"]
    assert out.loc["P1", "flag"] == ""                          # an ordinary patient is not flagged


def test_outliers_without_labels_only_scores_isolation():
    _, _, d = synthetic_cohort(n=200, seed=0)
    out = patient_outliers(d, k=10)
    assert "discordance" not in out.columns
    assert set(out["flag"]) <= {"", "isolated"}


def test_explain_outlier_names_terms_and_gaps():
    c = synthetic_hpo_cohort(n=200, seed=2)
    d = c.distance("cosine")
    labels = c.labels("diagnosis")
    worst = patient_outliers(d, labels, ids=c.ids, k=10).iloc[0]["id"]
    info = explain_outlier(c, c.ids.index(worst), d, labels, k=10)
    assert info["id"] == worst
    assert {"unusual_for_neighbourhood", "expected_but_absent", "neighbourhood_majority"} <= set(info)
    assert all(ic > 0 for _, _, ic in info["most_specific_terms"])      # uninformative terms dropped


def test_tiny_and_malformed_inputs_are_refused():
    with pytest.raises(ValueError):
        patient_outliers(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        patient_outliers(np.zeros((1, 1)))
