import numpy as np

from phenotopo import annotation_bias, phenotype_qc
from phenotopo.data import synthetic_hpo_cohort


def test_qc_flags_the_thin_site():
    c = synthetic_hpo_cohort(n=300, seed=1)
    qc = phenotype_qc(c)
    pat = qc["patients"].set_index("id")
    site = c.metadata["site"]
    thin = pat.loc[site[site == "B"].index, "n_terms"]
    thick = pat.loc[site[site == "A"].index, "n_terms"]
    assert thin.median() < thick.median()
    assert qc["summary"]["pct_review_recommended"] > 0
    assert (qc["flagged"]["flags"] != "").all()
    assert list(qc["patients"]["id"]) == list(c.ids)          # cohort order, alignable
    assert qc["flagged"].iloc[0]["n_terms"] == qc["patients"]["n_terms"].min()   # worst first


def test_flags_are_relative_and_recommend_review_not_a_verdict():
    c = synthetic_hpo_cohort(n=200, seed=1)
    qc = phenotype_qc(c)
    absolute = {f for row in qc["patients"]["flags_absolute"] for f in row.split("; ") if f}
    relative = {f for row in qc["patients"]["flags_relative"] for f in row.split("; ") if f}
    assert absolute <= {"NO_PHENOTYPE_RECORDED", "LOW_ANNOTATION_DEPTH", "HIGH_REDUNDANCY"}
    assert relative <= {"LOW_SPECIFICITY_RELATIVE_TO_COHORT", "LOW_DEPTH_RELATIVE_TO_COHORT"}
    assert not (absolute & relative)                             # the two layers stay separate
    assert not any("under" in f.lower() for f in absolute | relative)     # no absolute claim
    assert qc["summary"]["cohort_level_warning"] == ""           # this cohort is not thin overall
    assert set(qc["patients"]["review"]) <= {"", "review recommended"}
    assert (qc["patients"].loc[qc["patients"]["flags"] != "", "review"] == "review recommended").all()


def test_qc_counts_redundant_ancestors_and_missing_context():
    c = synthetic_hpo_cohort(n=300, seed=1)
    qc = phenotype_qc(c)
    assert qc["patients"]["n_redundant"].max() >= 1          # planted parent+child pairs
    assert qc["summary"]["pct_with_excluded_phenotypes"] == 0   # none recorded in this cohort
    assert qc["summary"]["pct_with_onset"] == 0


def test_annotation_bias_detects_a_site_confound():
    c = synthetic_hpo_cohort(n=300, seed=1)
    qc = phenotype_qc(c)
    bias = annotation_bias(qc, c.labels("site"), distance=c.distance("cosine"), k=10)
    assert bias["kruskal"]["n_terms"]["p"] < 0.05            # sites differ in annotation depth
    assert bias["kruskal"]["n_terms"]["epsilon_sq"] > 0.06
    r = bias["recovery"]
    assert r["metric"].startswith("balanced accuracy")
    assert 0 <= r["annotation_share_of_lift"] <= 1
    assert r["confound_risk"] in {"MODERATE", "HIGH"}        # by construction, depth tracks site
    assert "does not establish" in r["note"]                 # risk, never a causal verdict
    assert set(bias["by_group"].columns) == {"n", "median_terms", "median_mean_ic", "pct_flagged"}


def test_annotation_bias_is_quiet_when_groups_are_annotated_alike():
    c = synthetic_hpo_cohort(n=300, seed=1, thin_site=False)
    bias = annotation_bias(phenotype_qc(c), c.labels("diagnosis"),
                           distance=c.distance("cosine"), k=10)
    assert bias["kruskal"]["n_terms"]["p"] > 0.05
    assert bias["recovery"]["confound_risk"] == "LOW"


def test_a_uniformly_thin_cohort_is_called_out_even_though_nothing_stands_out():
    """Cohort-relative flags are silent when everybody is equally badly phenotyped."""
    from phenotopo import Cohort, Ontology

    onto = Ontology([("a", "root"), ("b", "root")], {t: t for t in ["a", "b", "root"]})
    c = Cohort(ids=[f"P{i}" for i in range(60)], present=[{"a"} if i % 2 else {"b"} for i in range(60)],
               ontology=onto)
    qc = phenotype_qc(c)
    assert qc["summary"]["cohort_level_warning"]                 # the absolute layer speaks up
    assert qc["summary"]["pct_flagged_absolute"] == 100.0


def test_k_is_bounded_by_the_smallest_training_fold():
    """A small cohort must not crash with the default k = 15."""
    from phenotopo import Cohort, Ontology

    onto = Ontology([("a", "root"), ("b", "root"), ("c", "root")], {t: t for t in ["a", "b", "c", "root"]})
    present = [{"a", "b"} if i % 2 else {"c"} for i in range(12)]
    labels = np.array(["X", "Y"] * 6)
    c = Cohort(ids=[f"P{i}" for i in range(12)], present=present, ontology=onto)
    bias = annotation_bias(phenotype_qc(c), labels, distance=c.distance(), k=15)
    assert bias["recovery"]["k"] < 15 and bias["recovery"]["k_requested"] == 15
    assert 0 <= bias["recovery"]["phenotype"] <= 1               # and it actually ran
