import matplotlib
matplotlib.use("Agg")

from phenotopo import group_connectivity, knn_graph, plot_connectivity, plot_density, plot_small_multiples
from phenotopo.data import synthetic_cohort


def test_plots_run_end_to_end():
    emb, labels, d = synthetic_cohort(n=400, seed=0)
    conn = group_connectivity(knn_graph(distance=d, k=10), labels)
    assert plot_connectivity(conn, embedding=emb, labels=labels) is not None
    assert plot_connectivity(conn) is not None          # spring layout fallback
    assert plot_density(emb, labels) is not None
    fig = plot_small_multiples(emb, labels, ncols=3)
    assert len(fig.axes) >= 5
