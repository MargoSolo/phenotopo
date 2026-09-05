import os

from phenotopo import cohort_report, connectivity_robustness, explain_groups
from phenotopo.data import synthetic_hpo_cohort


def test_report_is_one_self_contained_file(tmp_path):
    c = synthetic_hpo_cohort(n=150, seed=4)
    d = c.distance("cosine")
    comparison = explain_groups(c, c.labels("diagnosis"), "GENE_A", "GENE_B", n_perm=100, min_effect=0.15)
    rob = connectivity_robustness({"cosine": d, "simgic": c.distance("simgic")},
                                  c.labels("diagnosis"), ks=(10,), min_size=20,
                                  n_perm=50, n_boot=10)
    path = cohort_report(c, labels="diagnosis", distance=d, path=str(tmp_path / "r.html"),
                         robustness=rob, comparisons=comparison, title="Toy cohort")
    html = open(path, encoding="utf-8").read()
    assert os.path.exists(path) and html.startswith("<!doctype html>")
    assert "http://" not in html and "https://" not in html      # nothing is fetched from outside
    assert "data:image/png;base64," in html                       # figures embedded, not linked
    for section in ("Cohort overview", "Phenotyping quality", "Annotation bias",
                    "Outliers", "robustness", "GENE_A vs GENE_B"):
        assert section in html


def test_report_works_with_qc_alone(tmp_path):
    c = synthetic_hpo_cohort(n=80, seed=5)
    path = cohort_report(c, path=str(tmp_path / "qc_only.html"))
    html = open(path, encoding="utf-8").read()
    assert "Phenotyping quality" in html and "Outliers" not in html
