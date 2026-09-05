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
    assert qc["summary"]["pct_flagged"] > 0
    assert (qc["flagged"]["flags"] != "").all()
    assert list(qc["patients"]["id"]) == list(c.ids)          # cohort order, alignable
    assert qc["flagged"].iloc[0]["n_terms"] == qc["patients"]["n_terms"].min()   # worst first


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
    assert 0 <= r["share_explained_by_depth"] <= 1
    assert r["share_explained_by_depth"] > 0.25              # by construction, largely depth
    assert set(bias["by_group"].columns) == {"n", "median_terms", "median_mean_ic", "pct_flagged"}


def test_annotation_bias_is_quiet_when_groups_are_annotated_alike():
    c = synthetic_hpo_cohort(n=300, seed=1, thin_site=False)
    bias = annotation_bias(phenotype_qc(c), c.labels("diagnosis"),
                           distance=c.distance("cosine"), k=10)
    assert bias["kruskal"]["n_terms"]["p"] > 0.05
    assert bias["recovery"]["share_explained_by_depth"] < 0.5
