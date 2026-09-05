import numpy as np
import pytest

from phenotopo.data import synthetic_cohort


def test_mapper_graph_covers_points():
    pytest.importorskip("kmapper")
    from phenotopo.mapper import mapper_graph, node_values
    emb, labels, _ = synthetic_cohort(n=300, seed=0)
    _, g = mapper_graph(emb, n_cubes=6, perc_overlap=0.3)
    assert g.number_of_nodes() > 0
    covered = set()
    for n in g.nodes:
        covered |= set(g.nodes[n]["members"])
    assert len(covered) > 0.5 * len(emb)
    vals = node_values(g, np.arange(len(emb)))
    assert set(vals) == set(g.nodes)
