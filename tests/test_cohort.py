import json
import os

import numpy as np
import pandas as pd
import pytest

from phenotopo import Cohort, Ontology, from_hpo_table, from_phenopackets
from phenotopo.data import synthetic_hpo_cohort


def _toy_ontology():
    return Ontology([("c1", "p1"), ("c2", "p1"), ("p1", "root"), ("c3", "p2"), ("p2", "root")],
                    {t: t.upper() for t in ["c1", "c2", "c3", "p1", "p2", "root"]})


def test_ontology_ancestors_and_depth():
    o = _toy_ontology()
    assert o.ancestors("c1") == frozenset({"c1", "p1", "root"})
    assert o.ancestors("c1", include_self=False) == frozenset({"p1", "root"})
    assert (o.depth("root"), o.depth("p1"), o.depth("c1")) == (0, 1, 2)
    assert o.name("c1") == "C1" and o.name("HP:9") == "HP:9"


def test_propagation_ic_and_distances():
    c = Cohort(ids=["a", "b", "c"], present=[{"c1"}, {"c2"}, {"c3"}], ontology=_toy_ontology())
    assert c.propagated()[0] == {"c1", "p1", "root"}
    ic = c.information_content()
    assert ic["root"] == 0                                 # annotates everyone
    assert ic["c1"] > ic["p1"] > 0                          # rarer term is more informative
    for metric in ("cosine", "simgic"):
        d = c.distance(metric)
        assert d.shape == (3, 3) and np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
        assert d[0, 1] < d[0, 2]                            # shared parent beats no overlap
    with pytest.raises(ValueError):
        c.distance("euclidean")


def test_from_hpo_table_keeps_metadata_and_ignores_junk():
    df = pd.DataFrame({"id": [1, 2], "hpo": ["HP:0001250, HP:0002133", "n/a"], "grp": ["x", "y"]})
    c = from_hpo_table(df, terms_col="hpo")
    assert c.ids == ["1", "2"] and c.present[0] == {"HP:0001250", "HP:0002133"} and c.present[1] == set()
    assert list(c.labels("grp")) == ["x", "y"]


def test_from_phenopackets_reads_excluded_onset_and_gene(tmp_path):
    packet = {
        "subject": {"id": "P1", "sex": "FEMALE"},
        "phenotypicFeatures": [
            {"type": {"id": "HP:0001250", "label": "Seizure"}, "onset": {"age": {"iso8601duration": "P2Y"}}},
            {"type": {"id": "HP:0000252"}, "excluded": True},
        ],
        "interpretations": [{"diagnosis": {
            "disease": {"id": "OMIM:1", "label": "Toy disease"},
            "genomicInterpretations": [{"variantInterpretation": {
                "variationDescriptor": {"geneContext": {"symbol": "SCN2A"}}}}]}}],
    }
    (tmp_path / "p1.json").write_text(json.dumps(packet))
    c = from_phenopackets(str(tmp_path))
    assert c.ids == ["P1"] and c.present[0] == {"HP:0001250"} and c.excluded[0] == {"HP:0000252"}
    assert c.onset[0] == {"HP:0001250": "P2Y"}
    assert c.metadata.loc["P1", "gene"] == "SCN2A" and bool(c.metadata.loc["P1", "solved"])
    assert c.metadata.loc["P1", "disease"] == "Toy disease"


def test_synthetic_cohort_has_the_designed_faults():
    c = synthetic_hpo_cohort(n=200, seed=0)
    assert len(c) == 200 and set(c.metadata["diagnosis"]) == {"GENE_A", "GENE_B"}
    assert (c.metadata["site"] == "B").any()
    assert min(len(t) for t in c.present) == 1               # thin site
    assert c.distance("cosine").shape == (200, 200)


def test_length_mismatch_is_refused():
    with pytest.raises(ValueError):
        Cohort(ids=["a", "b"], present=[{"c1"}])
