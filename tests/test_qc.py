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
    flags = {f for row in qc["patients"]["flags"] for f in row.split("; ") if f}
    assert flags <= {"NO_PHENOTYPE_RECORDED", "LOW_ANNOTATION_DEPTH",
                     "LOW_SPECIFICITY_RELATIVE_TO_COHORT", "HIGH_REDUNDANCY"}
    assert not any("under" in f.lower() for f in flags)          # no absolute claim
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
